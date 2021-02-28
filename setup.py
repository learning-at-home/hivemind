import codecs
import glob
import os
import re

from pkg_resources import parse_requirements
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


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


class ProtoCompileInstall(install):
    def run(self):
        proto_compile(os.path.join(self.build_lib, 'hivemind', 'proto'))
        super().run()


class ProtoCompileDevelop(develop):
    def run(self):
        proto_compile(os.path.join('hivemind', 'proto'))
        super().run()


here = os.path.abspath(os.path.dirname(__file__))

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
    cmdclass={'install': ProtoCompileInstall, 'develop': ProtoCompileDevelop},
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
        'console_scripts': ['hivemind-server = scripts.run_server:main', ]
    },
    # What does your project relate to?
    keywords='pytorch, deep learning, machine learning, gpu, distributed computing, volunteer computing, dht',
)
