"""Genetic Programming in Python, with a scikit-learn inspired API

``gplearn`` is a set of algorithms for learning genetic programming models.

"""
# __version__ = '0.5.dev0'

from . import originalFeature
from . import dataload

__all__ = ['genetic', 'functions', 'fitness', 'originalFeature', 'dataload', 'oi_dataload']
