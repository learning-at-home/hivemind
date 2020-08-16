from pkg_resources import parse_requirements
from setuptools import setup
import codecs
import re
import os

here = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as requirements_file:
    install_requires = [str(requirement) for requirement in parse_requirements(requirements_file)]

# loading version from setup.py
with codecs.open(os.path.join(here, 'hivemind/__init__.py'), encoding='utf-8') as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    version_string = version_match.group(1)

setup(
    name='hivemind',
    version=version_string,
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
