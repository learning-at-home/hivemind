import codecs
import glob
import hashlib
import os
import platform
import re
import subprocess
import tarfile
import tempfile
import urllib.request

from pkg_resources import parse_requirements, parse_version
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

P2PD_VERSION = "v0.5.0.hivemind1"

P2PD_SOURCE_URL = f"https://github.com/learning-at-home/go-libp2p-daemon/archive/refs/tags/{P2PD_VERSION}.tar.gz"
P2PD_BINARY_URL = f"https://github.com/learning-at-home/go-libp2p-daemon/releases/download/{P2PD_VERSION}/"

# The value is sha256 of the binary from the release page
P2P_BINARY_HASH = {
    "p2pd-darwin-amd64": "fe00f9d79e8e4e4c007144d19da10b706c84187b3fb84de170f4664c91ecda80",
    "p2pd-darwin-arm64": "0404981a9c2b7cab5425ead2633d006c61c2c7ec85ac564ef69413ed470e65bd",
    "p2pd-linux-amd64": "42f8f48e62583b97cdba3c31439c08029fb2b9fc506b5bdd82c46b7cc1d279d8",
    "p2pd-linux-arm64": "046f18480c785a84bdf139d7486086d379397ca106cb2f0191598da32f81447a",
}

here = os.path.abspath(os.path.dirname(__file__))


def sha256(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def proto_compile(output_path):
    import grpc_tools.protoc

    cli_args = [
        "grpc_tools.protoc",
        "--proto_path=hivemind/proto",
        f"--python_out={output_path}",
    ] + glob.glob("hivemind/proto/*.proto")

    code = grpc_tools.protoc.main(cli_args)
    if code:  # hint: if you get this error in jupyter, run in console for richer error message
        raise ValueError(f"{' '.join(cli_args)} finished with exit code {code}")
    # Make pb2 imports in generated scripts relative
    for script in glob.iglob(f"{output_path}/*.py"):
        with open(script, "r+") as file:
            code = file.read()
            file.seek(0)
            file.write(re.sub(r"\n(import .+_pb2.*)", "from . \\1", code))
            file.truncate()


def _parse_platform(target_platform):
    """Parse the target platform and return Go build environment variables"""
    # Only allow platforms we have precompiled binaries for
    supported_platforms = ["linux-amd64", "linux-arm64", "darwin-amd64", "darwin-arm64"]
    if target_platform not in supported_platforms:
        raise ValueError(f"Unsupported platform: {target_platform}. Supported: {supported_platforms}")

    os_name, arch = target_platform.split("-", 1)
    return {"GOOS": os_name, "GOARCH": arch}


def _detect_current_platform():
    """Detect the current platform in the standard format"""
    arch = platform.machine()
    if arch in ("x86_64", "x64"):
        arch = "amd64"
    elif arch in ("aarch64", "aarch64_be", "armv8b", "armv8l"):
        arch = "arm64"
    return f"{platform.system().lower()}-{arch}"


def build_p2p_daemon(target_platform, output_dir):
    """Build p2pd from the source, optionally for a specific target platform

    Args:
        target_platform (str): Target platform in the format 'os-arch' (e.g., 'linux-amd64').
                                        If None, builds for the current platform.
        output_dir (str): Directory where the p2pd binary will be placed
    """
    result = subprocess.run("go version", capture_output=True, shell=True).stdout.decode("ascii", "replace")
    m = re.search(r"^go version go([\d.]+)", result)

    if m is None:
        raise FileNotFoundError("Could not find an installation of golang")
    version = parse_version(m.group(1))
    if version < parse_version("1.13"):
        raise OSError(f"Newer version of go required: must be >= 1.13, found {version}")

    env = os.environ.copy()

    if target_platform is None:
        target_platform = _detect_current_platform()

    env.update(_parse_platform(target_platform))
    print(f"Building p2pd for the current platform ({target_platform})")

    with tempfile.TemporaryDirectory() as tempdir:
        dest = os.path.join(tempdir, "libp2p-daemon.tar.gz")
        urllib.request.urlretrieve(P2PD_SOURCE_URL, dest)

        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(tempdir)

        output_path = os.path.abspath(os.path.join(output_dir, "hivemind", "hivemind_cli", "p2pd"))
        result = subprocess.run(
            ["go", "build", "-o", output_path],
            cwd=os.path.join(tempdir, f"go-libp2p-daemon-{P2PD_VERSION.lstrip('v')}", "p2pd"),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to build p2pd: exited with the status code: {result.returncode}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )


def download_p2p_daemon(target_platform=None, output_dir=here):
    """Download platform-specific p2pd binary

    Args:
        target_platform (str, optional): Target platform in the format 'os-arch' (e.g., 'linux-amd64')
        output_dir (str, optional): Directory where the p2pd binary will be placed. Defaults to source tree.
    """
    if target_platform is None:
        target_platform = _detect_current_platform()

    binary_name = f"p2pd-{target_platform}"

    if binary_name not in P2P_BINARY_HASH:
        raise RuntimeError(
            f"hivemind does not provide a precompiled p2pd binary for {target_platform}. "
            f"Please install Go and build it from source: https://github.com/learning-at-home/hivemind#from-source"
        )
    expected_hash = P2P_BINARY_HASH[binary_name]

    binary_path = os.path.join(str(output_dir), "hivemind", "hivemind_cli", "p2pd")
    if sha256(binary_path) != expected_hash:
        binary_url = f"{P2PD_BINARY_URL.rstrip('/')}/{binary_name}"
        print(f"Downloading {binary_url} to {binary_path}")

        urllib.request.urlretrieve(binary_url, binary_path)
        os.chmod(binary_path, 0o777)

        actual_hash = sha256(binary_path)
        if actual_hash != expected_hash:
            os.unlink(binary_path)
            raise RuntimeError(
                f"The sha256 checksum for p2pd does not match (expected: {expected_hash}, actual: {actual_hash})"
            )

        print(f"Downloaded {binary_name}")


class BuildPy(build_py):
    editable_mode = False

    def initialize_options(self):
        super().initialize_options()
        self.editable_mode = False

    def run(self):
        target_platform = os.environ.get("HIVEMIND_TARGET_PLATFORM")
        buildgo = os.environ.get("HIVEMIND_BUILDGO", "").lower() in ("1", "true", "yes")

        if self.editable_mode:
            output_dir = here
        else:
            super().run()
            output_dir = self.build_lib

        if buildgo:
            build_p2p_daemon(target_platform=target_platform, output_dir=output_dir)
        else:
            download_p2p_daemon(target_platform=target_platform, output_dir=output_dir)

        proto_compile(os.path.join(output_dir, "hivemind", "proto"))

    def get_output_mapping(self):
        mapping = {}
        if hasattr(super(), "get_output_mapping"):
            mapping = super().get_output_mapping()

        proto_dir = "hivemind/proto"
        for proto_file in glob.glob(os.path.join(proto_dir, "*.proto")):
            pb2_name = os.path.basename(proto_file).replace(".proto", "_pb2.py")
            if self.editable_mode:
                dest = os.path.join(proto_dir, pb2_name)
                mapping[dest] = dest
            else:
                dest = os.path.join(self.build_lib, proto_dir, pb2_name)
                mapping[dest] = os.path.join(proto_dir, pb2_name)

        return mapping


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


with open("requirements.txt") as requirements_file:
    install_requires = list(map(str, parse_requirements(requirements_file)))

# loading version from setup.py
with codecs.open(os.path.join(here, "hivemind", "__init__.py"), encoding="utf-8") as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.MULTILINE)
    version_string = version_match.group(1)

extras = {}

with open("requirements-dev.txt") as dev_requirements_file:
    extras["dev"] = list(map(str, parse_requirements(dev_requirements_file)))

with open("requirements-docs.txt") as docs_requirements_file:
    extras["docs"] = list(map(str, parse_requirements(docs_requirements_file)))

extras["bitsandbytes"] = ["bitsandbytes~=0.45.2"]

extras["all"] = extras["dev"] + extras["docs"] + extras["bitsandbytes"]

setup(
    name="hivemind",
    version=version_string,
    cmdclass={"build_py": BuildPy},
    distclass=BinaryDistribution,
    description="Decentralized deep learning in PyTorch",
    long_description="Decentralized deep learning in PyTorch. Built to train models on thousands of volunteers "
    "across the world.",
    author="Learning@home & contributors",
    author_email="hivemind-team@hotmail.com",
    url="https://github.com/learning-at-home/hivemind",
    packages=find_packages(exclude=["tests"]),
    package_data={"hivemind": ["proto/*", "hivemind_cli/*"]},
    include_package_data=True,
    license="MIT",
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "hivemind-dht = hivemind.hivemind_cli.run_dht:main",
            "hivemind-server = hivemind.hivemind_cli.run_server:main",
        ]
    },
    # What does your project relate to?
    keywords="pytorch, deep learning, machine learning, gpu, distributed computing, volunteer computing, dht",
)
