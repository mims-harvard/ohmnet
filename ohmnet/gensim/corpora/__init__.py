"""
This package contains implementations of various streaming corpus I/O format.
"""

# bring corpus classes directly into package namespace, to save some typing
from .indexedcorpus import IndexedCorpus # must appear before the other classes

from .dictionary import Dictionary
