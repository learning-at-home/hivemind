import codecs
import glob
import hashlib
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import urllib.request

from pkg_resources import parse_requirements, parse_version
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

P2PD_VERSION = "v0.3.6"
P2PD_CHECKSUM = "627d0c3b475a29331fdfd1667e828f6d"
LIBP2P_TAR_URL = f"https://github.com/learning-at-home/go-libp2p-daemon/archive/refs/tags/{P2PD_VERSION}.tar.gz"
P2PD_BINARY_URL = f"https://github.com/learning-at-home/go-libp2p-daemon/releases/download/{P2PD_VERSION}/p2pd"

here = os.path.abspath(os.path.dirname(__file__))


def md5(fname, chunk_size=4096):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def proto_compile(output_path):
    import grpc_tools.protoc

    cli_args = [
        "grpc_tools.protoc",
        "--proto_path=hivemind/proto",
        f"--python_out={output_path}",
        f"--grpc_python_out={output_path}",
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
        urllib.request.urlretrieve(LIBP2P_TAR_URL, dest)

        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(tempdir)

        result = subprocess.run(
            f'go build -o {shlex.quote(os.path.join(here, "hivemind", "hivemind_cli", "p2pd"))}',
            cwd=os.path.join(tempdir, f"go-libp2p-daemon-{P2PD_VERSION[1:]}", "p2pd"),
            shell=True,
        )

        if result.returncode:
            raise RuntimeError(
                "Failed to build or install libp2p-daemon:" f" exited with status code: {result.returncode}"
            )


def download_p2p_daemon():
    install_path = os.path.join(here, "hivemind", "hivemind_cli")
    binary_path = os.path.join(install_path, "p2pd")
    if not os.path.exists(binary_path) or md5(binary_path) != P2PD_CHECKSUM:
        print("Downloading Peer to Peer Daemon")
        urllib.request.urlretrieve(P2PD_BINARY_URL, binary_path)
        os.chmod(binary_path, 0o777)
        if md5(binary_path) != P2PD_CHECKSUM:
            raise RuntimeError(f"Downloaded p2pd binary from {P2PD_BINARY_URL} does not match with md5 checksum")


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

extras["all"] = extras["dev"] + extras["docs"]

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
            "hivemind-server = hivemind.hivemind_cli.run_server:main",
        ]
    },
    # What does your project relate to?
    keywords="pytorch, deep learning, machine learning, gpu, distributed computing, volunteer computing, dht",
)
