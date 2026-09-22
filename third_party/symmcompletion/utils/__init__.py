"""
Vendored SymmCompletion utility package.

This file intentionally makes the upstream utils/ directory an explicit
Python package so that historical imports such as `from utils import registry`
resolve to the vendored SymmCompletion implementation when its repository
root is placed first on sys.path.
"""
