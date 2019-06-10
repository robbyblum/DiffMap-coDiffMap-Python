# DiffMap and coDiffMap, Python implementation
DiffMap and coDiffMap Python code. Core Python DiffMap functionality written by Robert Blum; coDiffMap adaptation written by Jared Rovny.


This code is able to replicate the analysis done in Igor Pro for our forthcoming paper in the Journal of Biomolecular NMR (link will be added when published). We have also designed it to work with [nmrglue](https://www.nmrglue.com/), for better cross-compatibility.

The code assumes that your data starts as a numpy ndarray (such as the ones created by nmrglue data load functions). If your data starts in a hypercomplex States-like format, as the data from our paper did, we have included a function to convert it into the "MRI-like" format necessary for our method.

## Requirements
Written in Python 3; requires numpy.

## Installation
For now, you can download the source code and add the python files to your PYTHONPATH.

## Caveats
This code is a work in progress, and likely doesn't cover all the cases that one might want to use it with. Proper installation methods and proper dependency checks and the like aren't finished yet. We will hopefully be able to clean these things up soon, but we have to work around dissertating for now. If you want to use the code and aren't sure where to begin, or have any other questions about it, open an issue and ask!
