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
from setuptools.command.develop import develop

P2PD_VERSION = "v0.3.18"

P2PD_SOURCE_URL = f"https://github.com/learning-at-home/go-libp2p-daemon/archive/refs/tags/{P2PD_VERSION}.tar.gz"
P2PD_BINARY_URL = f"https://github.com/learning-at-home/go-libp2p-daemon/releases/download/{P2PD_VERSION}/"

# The value is sha256 of the binary from the release page
P2P_BINARY_HASH = {
    "p2pd-darwin-amd64": "a9e5fee6bdcbfb5cc7f1a9b19e3fa4c91ceb18108b20472bf7affa62c590a964",
    "p2pd-darwin-arm64": "5868f2000f4d0746c5349f3480cb5450f66d4391570b420a8f888afc9e9152af",
    "p2pd-linux-amd64": "a9d456006e915d3cbbc8e5e9902d5b1a3120235f317e113137d6b631e2c810ac",
    "p2pd-linux-arm64": "b03c02e6afa47f158183f6267eecf4adb0e17120d534402c99dc33f3a2915cb1",
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


def build_p2p_daemon():
    result = subprocess.run("go version", capture_output=True, shell=True).stdout.decode("ascii", "replace")
    m = re.search(r"^go version go([\d.]+)", result)

    if m is None:
        raise FileNotFoundError("Could not find golang installation")
    version = parse_version(m.group(1))
    if version < parse_version("1.13"):
        raise EnvironmentError(f"Newer version of go required: must be >= 1.13, found {version}")

    with tempfile.TemporaryDirectory() as tempdir:
        dest = os.path.join(tempdir, "libp2p-daemon.tar.gz")
        urllib.request.urlretrieve(P2PD_SOURCE_URL, dest)

        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(tempdir)

        result = subprocess.run(
            ["go", "build", "-o", os.path.join(here, "hivemind", "hivemind_cli", "p2pd")],
            cwd=os.path.join(tempdir, f"go-libp2p-daemon-{P2PD_VERSION.lstrip('v')}", "p2pd"),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build p2pd: exited with status code: {result.returncode}")


def download_p2p_daemon():
    binary_path = os.path.join(here, "hivemind", "hivemind_cli", "p2pd")
    arch = platform.machine()
    # An architecture name may vary depending on the OS (e.g., the same CPU is arm64 on macOS and aarch64 on Linux).
    # We consider multiple aliases here, see https://stackoverflow.com/questions/45125516/possible-values-for-uname-m
    if arch in ("x86_64", "x64"):
        arch = "amd64"
    if arch in ("aarch64", "aarch64_be", "armv8b", "armv8l"):
        arch = "arm64"
    binary_name = f"p2pd-{platform.system().lower()}-{arch}"

    if binary_name not in P2P_BINARY_HASH:
        raise RuntimeError(
            f"hivemind does not provide a precompiled p2pd binary for {platform.system()} ({arch}). "
            f"Please install Go and build it from source: https://github.com/learning-at-home/hivemind#from-source"
        )
    expected_hash = P2P_BINARY_HASH[binary_name]

    if sha256(binary_path) != expected_hash:
        binary_url = os.path.join(P2PD_BINARY_URL, binary_name)
        print(f"Downloading {binary_url}")

        urllib.request.urlretrieve(binary_url, binary_path)
        os.chmod(binary_path, 0o777)

        actual_hash = sha256(binary_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"The sha256 checksum for p2pd does not match (expected: {expected_hash}, actual: {actual_hash})"
            )


class BuildPy(build_py):
    user_options = build_py.user_options + [("buildgo", None, "Builds p2pd from source")]

    def initialize_options(self):
        super().initialize_options()
        self.buildgo = False

    def run(self):
        if self.buildgo:
            build_p2p_daemon()
        else:
            download_p2p_daemon()

        super().run()

        proto_compile(os.path.join(self.build_lib, "hivemind", "proto"))


class Develop(develop):
    def run(self):
        self.reinitialize_command("build_py", build_lib=here)
        self.run_command("build_py")
        super().run()


with open("requirements.txt") as requirements_file:
    install_requires = list(map(str, parse_requirements(requirements_file)))

# loading version from setup.py
with codecs.open(os.path.join(here, "hivemind/__init__.py"), encoding="utf-8") as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    version_string = version_match.group(1)

extras = {}

with open("requirements-dev.txt") as dev_requirements_file:
    extras["dev"] = list(map(str, parse_requirements(dev_requirements_file)))

with open("requirements-docs.txt") as docs_requirements_file:
    extras["docs"] = list(map(str, parse_requirements(docs_requirements_file)))

extras["bitsandbytes"] = ["bitsandbytes~=0.37.0"]

extras["all"] = extras["dev"] + extras["docs"] + extras["bitsandbytes"]

setup(
    name="hivemind",
    version=version_string,
    cmdclass={"build_py": BuildPy, "develop": Develop},
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
    setup_requires=["grpcio-tools"],
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
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
