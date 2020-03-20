# setup script for diffmap package. NOT DONE! Currently mostly the same as the
# package tutorial: https://packaging.python.org/tutorials/packaging-projects/

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DiffMap-coDiffMap-Python",
    version="0.1.0-dev",
    author="Robby Blum",
    author_email="robbyblum@gmail.com",
    description="Python DiffMap and coDiffMap code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/robbyblum/DiffMap-coDiffMap-Python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)


# below is nmrglue's setup.py code, for comparison. Later I may add more stuff
# to the actual code which might make it look more like below...

# #!/usr/bin/env python
#
# # setup script for nmrglue
#
# from setuptools import setup
# from codecs import open
# from os import path, walk
#
# here = path.abspath(path.dirname(__file__))
#
# # get long description from README
# with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
#     long_description = f.read()
#
# setup(
#     name='nmrglue',
#     version='0.8-dev',  # change this in nmrglue/__init__.py also
#     description='A module for working with NMR data in Python',
#     long_description=long_description,
#     url='http://www.nmrglue.com',
#     author='Jonathan J. Helmus',
#     author_email='jjhelmus@gmail.com',
#     license='New BSD License',
#     classifiers=[
#         'Intended Audience :: Science/Research',
#         'Intended Audience :: Developers',
#         'License :: OSI Approved :: BSD License',
#         'Programming Language :: Python :: 2',
#         'Programming Language :: Python :: 2.7',
#         'Programming Language :: Python :: 3',
#         'Programming Language :: Python :: 3.4',
#         'Programming Language :: Python :: 3.5',
#         'Programming Language :: Python :: 3.6',
#         'Programming Language :: Python :: 3.7',
#         'Topic :: Scientific/Engineering',
#         'Operating System :: MacOS :: MacOS X',
#         'Operating System :: Microsoft :: Windows',
#         'Operating System :: POSIX :: Linux'],
#     install_requires=['numpy', 'scipy'],
#     packages=[
#         'nmrglue',
#         'nmrglue.analysis',
#         'nmrglue.analysis.tests',
#         'nmrglue.fileio',
#         'nmrglue.fileio.tests',
#         'nmrglue.process',
#         'nmrglue.process.nmrtxt',
#         'nmrglue.util'],
#     include_package_data=True,
# )
