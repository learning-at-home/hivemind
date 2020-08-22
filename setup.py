import codecs
import glob
import os
import re

import grpc_tools.protoc
from pkg_resources import parse_requirements
from setuptools import setup
from setuptools.command.build_ext import build_ext

here = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as requirements_file:
    install_requires = [str(requirement) for requirement in parse_requirements(requirements_file)]

# loading version from setup.py
with codecs.open(os.path.join(here, 'hivemind/__init__.py'), encoding='utf-8') as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    version_string = version_match.group(1)




class ProtoCompileBuild(build_ext):
    def run(self):
        super().run()

        cli_args = ['grpc_tools.protoc',
                    '--proto_path=hivemind/proto', '--python_out=hivemind/proto',
                    '--grpc_python_out=hivemind/proto'] + glob.glob('hivemind/proto/*.proto')

        code = grpc_tools.protoc.main(cli_args)

        if code:  # hint: if you get this error in jupyter, run in console for richer error message
            raise ValueError(f"{' '.join(cli_args)} finished with exit code {code}")

        # Make pb2 imports in generated scripts relative
        for script in glob.iglob('hivemind/proto/*.py'):
            with open(script, 'r+') as file:
                code = file.read()
                file.seek(0)
                file.write(re.sub(r'\n(import .+_pb2.*)', 'from . \\1', code))
                file.truncate()


setup(
    name='hivemind',
    version=version_string,
    cmdclass={'build_ext': ProtoCompileBuild},
    description='',
    long_description='',
    author='Learning@home authors',
    author_email='mryabinin@hse.ru',
    url="https://github.com/learning-at-home/hivemind",
    packages=['hivemind'],
    license='MIT',
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    # What does your project relate to?
    keywords='pytorch, deep learning, machine learning, gpu, distributed computing',
)
