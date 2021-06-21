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
from setuptools.command.develop import develop
from setuptools.command.install import install

P2PD_VERSION = 'v0.3.1'
P2PD_CHECKSUM = '15292b880c6b31f5b3c36084b3acc17f'
LIBP2P_TAR_URL = f'https://github.com/learning-at-home/go-libp2p-daemon/archive/refs/tags/{P2PD_VERSION}.tar.gz'

here = os.path.abspath(os.path.dirname(__file__))


def md5(fname, chunk_size=4096):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def proto_compile(output_path):
    import grpc_tools.protoc

    cli_args = ['grpc_tools.protoc',
                '--proto_path=hivemind/proto', f'--python_out={output_path}',
                f'--grpc_python_out={output_path}'] + glob.glob('hivemind/proto/*.proto')

    code = grpc_tools.protoc.main(cli_args)
    if code:  # hint: if you get this error in jupyter, run in console for richer error message
        raise ValueError(f"{' '.join(cli_args)} finished with exit code {code}")
    # Make pb2 imports in generated scripts relative
    for script in glob.iglob(f'{output_path}/*.py'):
        with open(script, 'r+') as file:
            code = file.read()
            file.seek(0)
            file.write(re.sub(r'\n(import .+_pb2.*)', 'from . \\1', code))
            file.truncate()


def libp2p_build_install():
    result = subprocess.run("go version", capture_output=True, shell=True).stdout.decode('ascii', 'replace')
    m = re.search(r'^go version go([\d.]+)', result)

    if m is None:
        raise FileNotFoundError('Could not find golang installation')
    if parse_version(m.group(1)) < parse_version("1.13"):
        raise EnvironmentError(f'Newer version of go required: must be >= 1.13, found {version}')

    with tempfile.TemporaryDirectory() as tempdir:
        dest = os.path.join(tempdir, 'libp2p-daemon.tar.gz')
        urllib.request.urlretrieve(LIBP2P_TAR_URL, dest)

        with tarfile.open(dest, 'r:gz') as tar:
            tar.extractall(tempdir)

        result = subprocess.run(f'go build -o {shlex.quote(os.path.join(here, "hivemind", "hivemind_cli", "p2pd"))}',
                                cwd=os.path.join(tempdir, f'go-libp2p-daemon-{P2PD_VERSION[1:]}', 'p2pd'), shell=True)

        if result.returncode:
            raise RuntimeError('Failed to build or install libp2p-daemon:'
                               f' exited with status code: {result.returncode}')


def libp2p_download_install():
    install_path = os.path.join(here, 'hivemind', 'hivemind_cli')
    binary_path = os.path.join(install_path, 'p2pd')
    if not os.path.exists(binary_path) or md5(binary_path) != P2PD_CHECKSUM:
        print('Downloading Peer to Peer Daemon')
        url = f'https://github.com/learning-at-home/go-libp2p-daemon/releases/download/{P2PD_VERSION}/p2pd'
        urllib.request.urlretrieve(url, binary_path)
        os.chmod(binary_path, 0o777)
        if md5(binary_path) != P2PD_CHECKSUM:
            raise RuntimeError(f'Downloaded p2pd binary from {url} does not match with md5 checksum')


class Install(install):
    user_options = install.user_options + [('buildgo', None, "Builds p2pd from source")]

    def initialize_options(self):
        super().initialize_options()
        self.buildgo = False

    def run(self):
        if self.buildgo:
            libp2p_build_install()
        else:
            libp2p_download_install()
        proto_compile(os.path.join(self.build_lib, 'hivemind', 'proto'))
        super().run()


class Develop(develop):
    user_options = develop.user_options + [('buildgo', None, None)]

    def initialize_options(self):
        super().initialize_options()
        self.buildgo = False

    def run(self):
        if self.buildgo:
            libp2p_build_install()
        else:
            libp2p_download_install()
        proto_compile(os.path.join('hivemind', 'proto'))
        super().run()


with open('requirements.txt') as requirements_file:
    install_requires = list(map(str, parse_requirements(requirements_file)))

# loading version from setup.py
with codecs.open(os.path.join(here, 'hivemind/__init__.py'), encoding='utf-8') as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    version_string = version_match.group(1)

extras = {}

with open('requirements-dev.txt') as dev_requirements_file:
    extras['dev'] = list(map(str, parse_requirements(dev_requirements_file)))

with open('requirements-docs.txt') as docs_requirements_file:
    extras['docs'] = list(map(str, parse_requirements(docs_requirements_file)))

extras['all'] = extras['dev'] + extras['docs']

setup(
    name='hivemind',
    version=version_string,
    cmdclass={'install': Install, 'develop': Develop},
    description='Decentralized deep learning in PyTorch',
    long_description='Decentralized deep learning in PyTorch. Built to train giant models on '
                     'thousands of volunteers across the world.',
    author='Learning@home & contributors',
    author_email='mryabinin0@gmail.com',
    url="https://github.com/learning-at-home/hivemind",
    packages=find_packages(exclude=['tests']),
    package_data={'hivemind': ['proto/*']},
    include_package_data=True,
    license='MIT',
    setup_requires=['grpcio-tools'],
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    entry_points={
        'console_scripts': ['hivemind-server = hivemind.hivemind_cli.run_server:main', ]
    },
    # What does your project relate to?
    keywords='pytorch, deep learning, machine learning, gpu, distributed computing, volunteer computing, dht',
)
