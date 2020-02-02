from setuptools import setup, find_packages
import os


def read(fname, lines=False):
    with open(os.path.join(os.path.dirname(__file__), fname), 'r') as fd:
        return fd.readlines() if lines else fd.read()


setup(
    name='ACOio',
    version='0.2.2',
    description='IO tools for the Aloha Cabled Observaroty',
    packages=find_packages('.'),
    py_modules=['aco'],
    package_dirs={'aco':'aco'},
    python_requires='>=3.6',
    classifiers=[
        # Language suppoer
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",

        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Operatin System
        "Operating System :: OS Independent"
    ],

    # Dependencies
    install_requires=read('requirements.txt', True),

    # Description
    long_description=read('README.md'),
    long_description_content_type='text/markdown',

    # Meta information
    url="https://github.com/probinso/ACOio",
    author="Philip Robinson",
    author_email="probinso+acoio@protonmail.com"
)
