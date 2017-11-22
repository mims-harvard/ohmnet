#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This OhmNet code is adapted from:
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from __future__ import with_statement

import logging
import itertools

logger = logging.getLogger(__name__)

try:
    from html.entities import name2codepoint as n2cp
except ImportError:
    from htmlentitydefs import name2codepoint as n2cp
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

import sys
import os

import numpy
import scipy.sparse

if sys.version_info[0] >= 3:
    unicode = str

from six import iteritems

try:
    from smart_open import smart_open
except ImportError:
    logger.info("smart_open library not found; falling back to local-filesystem-only")

    def make_closing(base, **attrs):
        """
        Add support for `with Base(attrs) as fout:` to the base class if it's missing.
        The base class' `close()` method will be called on context exit, to always close the file properly.

        This is needed for gzip.GzipFile, bz2.BZ2File etc in older Pythons (<=2.6), which otherwise
        raise "AttributeError: GzipFile instance has no attribute '__exit__'".

        """
        if not hasattr(base, '__enter__'):
            attrs['__enter__'] = lambda self: self
        if not hasattr(base, '__exit__'):
            attrs['__exit__'] = lambda self, type, value, traceback: self.close()
        return type('Closing' + base.__name__, (base, object), attrs)

    def smart_open(fname, mode='rb'):
        _, ext = os.path.splitext(fname)
        if ext == '.bz2':
            from bz2 import BZ2File
            return make_closing(BZ2File)(fname, mode)
        if ext == '.gz':
            from gzip import GzipFile
            return make_closing(GzipFile)(fname, mode)
        return open(fname, mode)


def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')
to_utf8 = any2utf8


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)
to_unicode = any2unicode


class SaveLoad(object):
    """
    Objects which inherit from this class have save/load functions, which un/pickle
    them to disk.

    This uses pickle for de/serializing, so objects must not contain
    unpicklable attributes, such as lambda functions etc.

    """
    @classmethod
    def load(cls, fname, mmap=None):
        """
        Load a previously saved object from file (also see `save`).

        If the object was saved with large arrays stored separately, you can load
        these arrays via mmap (shared memory) using `mmap='r'`. Default: don't use
        mmap, load large arrays as normal objects.

        If the file being loaded is compressed (either '.gz' or '.bz2'), then
        `mmap=None` must be set.  Load will raise an `IOError` if this condition
        is encountered.

        """
        logger.info("loading %s object from %s" % (cls.__name__, fname))

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        obj = unpickle(fname)
        obj._load_specials(fname, mmap, compress, subname)
        return obj


    def _load_specials(self, fname, mmap, compress, subname):
        """
        Loads any attributes that were stored specially, and gives the same
        opportunity to recursively included SaveLoad instances.

        """

        mmap_error = lambda x, y: IOError(
            'Cannot mmap compressed object %s in file %s. ' % (x, y) +
            'Use `load(fname, mmap=None)` or uncompress files manually.')

        for attrib in getattr(self, '__recursive_saveloads', []):
            cfname = '.'.join((fname, attrib))
            logger.info("loading %s recursively from %s.* with mmap=%s" % (
                attrib, cfname, mmap))
            getattr(self, attrib)._load_specials(cfname, mmap, compress, subname)

        for attrib in getattr(self, '__numpys', []):
            logger.info("loading %s from %s with mmap=%s" % (
                attrib, subname(fname, attrib), mmap))

            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                val = numpy.load(subname(fname, attrib))['val']
            else:
                val = numpy.load(subname(fname, attrib), mmap_mode=mmap)

            setattr(self, attrib, val)

        for attrib in getattr(self, '__scipys', []):
            logger.info("loading %s from %s with mmap=%s" % (
                attrib, subname(fname, attrib), mmap))
            sparse = unpickle(subname(fname, attrib))
            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                with numpy.load(subname(fname, attrib, 'sparse')) as f:
                    sparse.data = f['data']
                    sparse.indptr = f['indptr']
                    sparse.indices = f['indices']
            else:
                sparse.data = numpy.load(subname(fname, attrib, 'data'), mmap_mode=mmap)
                sparse.indptr = numpy.load(subname(fname, attrib, 'indptr'), mmap_mode=mmap)
                sparse.indices = numpy.load(subname(fname, attrib, 'indices'), mmap_mode=mmap)

            setattr(self, attrib, sparse)

        for attrib in getattr(self, '__ignoreds', []):
            logger.info("setting ignored attribute %s to None" % (attrib))
            setattr(self, attrib, None)


    @staticmethod
    def _adapt_by_suffix(fname):
        """Give appropriate compress setting and filename formula"""
        if fname.endswith('.gz') or fname.endswith('.bz2'):
            compress = True
            subname = lambda *args: '.'.join(list(args) + ['npz'])
        else:
            compress = False
            subname = lambda *args: '.'.join(list(args) + ['npy'])
        return (compress, subname)


    def _smart_save(self, fname, separately=None, sep_limit=10 * 1024**2,
                    ignore=frozenset(), pickle_protocol=2):
        """
        Save the object to file (also see `load`).

        If `separately` is None, automatically detect large
        numpy/scipy.sparse arrays in the object being stored, and store
        them into separate files. This avoids pickle memory errors and
        allows mmap'ing large arrays back on load efficiently.

        You can also set `separately` manually, in which case it must be
        a list of attribute names to be stored in separate files. The
        automatic check is not performed in this case.

        `ignore` is a set of attribute names to *not* serialize (file
        handles, caches etc). On subsequent load() these attributes will
        be set to None.

        `pickle_protocol` defaults to 2 so the pickled object can be imported
        in both Python 2 and 3.

        """
        logger.info(
            "saving %s object under %s, separately %s" % (
                self.__class__.__name__, fname, separately))

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        restores = self._save_specials(fname, separately, sep_limit, ignore, pickle_protocol,
                                       compress, subname)
        try:
            pickle(self, fname, protocol=pickle_protocol)
        finally:
            # restore attribs handled specially
            for obj, asides in restores:
                for attrib, val in iteritems(asides):
                    setattr(obj, attrib, val)


    def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
        """
        Save aside any attributes that need to be handled separately, including
        by recursion any attributes that are themselves SaveLoad instances.

        Returns a list of (obj, {attrib: value, ...}) settings that the caller
        should use to restore each object's attributes that were set aside
        during the default pickle().

        """
        asides = {}
        sparse_matrices = (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)
        if separately is None:
            separately = []
            for attrib, val in iteritems(self.__dict__):
                if isinstance(val, numpy.ndarray) and val.size >= sep_limit:
                    separately.append(attrib)
                elif isinstance(val, sparse_matrices) and val.nnz >= sep_limit:
                    separately.append(attrib)

        # whatever's in `separately` or `ignore` at this point won't get pickled
        for attrib in separately + list(ignore):
            if hasattr(self, attrib):
                asides[attrib] = getattr(self, attrib)
                delattr(self, attrib)

        recursive_saveloads = []
        restores = []
        for attrib, val in iteritems(self.__dict__):
            if hasattr(val, '_save_specials'):  # better than 'isinstance(val, SaveLoad)' if IPython reloading
                recursive_saveloads.append(attrib)
                cfname = '.'.join((fname,attrib))
                restores.extend(val._save_specials(cfname, None, sep_limit, ignore,
                                                   pickle_protocol, compress, subname))

        try:
            numpys, scipys, ignoreds = [], [], []
            for attrib, val in iteritems(asides):
                if isinstance(val, numpy.ndarray) and attrib not in ignore:
                    numpys.append(attrib)
                    logger.info("storing numpy array '%s' to %s" % (
                        attrib, subname(fname, attrib)))

                    if compress:
                        numpy.savez_compressed(subname(fname, attrib), val=numpy.ascontiguousarray(val))
                    else:
                        numpy.save(subname(fname, attrib), numpy.ascontiguousarray(val))

                elif isinstance(val, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)) and attrib not in ignore:
                    scipys.append(attrib)
                    logger.info("storing scipy.sparse array '%s' under %s" % (
                        attrib, subname(fname, attrib)))

                    if compress:
                        numpy.savez_compressed(subname(fname, attrib, 'sparse'),
                                               data=val.data,
                                               indptr=val.indptr,
                                               indices=val.indices)
                    else:
                        numpy.save(subname(fname, attrib, 'data'), val.data)
                        numpy.save(subname(fname, attrib, 'indptr'), val.indptr)
                        numpy.save(subname(fname, attrib, 'indices'), val.indices)

                    data, indptr, indices = val.data, val.indptr, val.indices
                    val.data, val.indptr, val.indices = None, None, None

                    try:
                        # store array-less object
                        pickle(val, subname(fname, attrib), protocol=pickle_protocol)
                    finally:
                        val.data, val.indptr, val.indices = data, indptr, indices
                else:
                    logger.info("not storing attribute %s" % (attrib))
                    ignoreds.append(attrib)

            self.__dict__['__numpys'] = numpys
            self.__dict__['__scipys'] = scipys
            self.__dict__['__ignoreds'] = ignoreds
            self.__dict__['__recursive_saveloads'] = recursive_saveloads
        except:
            # restore the attributes if exception-interrupted
            for attrib, val in iteritems(asides):
                setattr(self, attrib, val)
            raise
        return restores + [(self, asides)]


    def save(self, fname_or_handle, separately=None, sep_limit=10 * 1024**2,
             ignore=frozenset(), pickle_protocol=2):
        """
        Save the object to file (also see `load`).

        `fname_or_handle` is either a string specifying the file name to
        save to, or an open file-like object which can be written to. If
        the object is a file handle, no special array handling will be
        performed; all attributes will be saved to the same file.

        If `separately` is None, automatically detect large
        numpy/scipy.sparse arrays in the object being stored, and store
        them into separate files. This avoids pickle memory errors and
        allows mmap'ing large arrays back on load efficiently.

        You can also set `separately` manually, in which case it must be
        a list of attribute names to be stored in separate files. The
        automatic check is not performed in this case.

        `ignore` is a set of attribute names to *not* serialize (file
        handles, caches etc). On subsequent load() these attributes will
        be set to None.

        `pickle_protocol` defaults to 2 so the pickled object can be imported
        in both Python 2 and 3.

        """
        try:
            _pickle.dump(self, fname_or_handle, protocol=pickle_protocol)
            logger.info("saved %s object" % self.__class__.__name__)
        except TypeError:  # `fname_or_handle` does not have write attribute
            self._smart_save(fname_or_handle, separately, sep_limit, ignore,
                             pickle_protocol=pickle_protocol)
#endclass SaveLoad

def pickle(obj, fname, protocol=2):
    """Pickle object `obj` to file `fname`.
    `protocol` defaults to 2 so pickled objects are compatible across
    Python 2.x and 3.x.
    """
    with open(fname, 'wb') as fout:  # 'b' for binary, needed on Windows
        _pickle.dump(obj, fout, protocol=protocol)


def unpickle(fname):
    """Load pickled object from `fname`"""
    with open(fname) as f:
        return _pickle.loads(f.read())


def prune_vocab(vocab, min_reduce, trim_rule=None):
    """
    Remove all entries from the `vocab` dictionary with count smaller than `min_reduce`.

    Modifies `vocab` in place, returns the sum of all counts that were pruned.

    """
    result = 0
    old_len = len(vocab)
    for w in list(vocab):  # make a copy of dict's keys
        if not keep_vocab_item(w, vocab[w], min_reduce, trim_rule):  # vocab[w] <= min_reduce:
            result += vocab[w]
            del vocab[w]
    logger.info("pruned out %i tokens with count <=%i (before %i, after %i)",
                old_len - len(vocab), min_reduce, old_len, len(vocab))
    return result


def qsize(queue):
    """Return the (approximate) queue size where available; -1 where not (OS X)."""
    try:
        return queue.qsize()
    except NotImplementedError:
        # OS X doesn't support qsize
        return -1


RULE_DEFAULT = 0
RULE_DISCARD = 1
RULE_KEEP = 2


def keep_vocab_item(word, count, min_count, trim_rule=None):
    default_res = count >= min_count

    if trim_rule is None:
        return default_res
    else:
        rule_res = trim_rule(word, count, min_count)
        if rule_res == RULE_KEEP:
            return True
        elif rule_res == RULE_DISCARD:
            return False
        else:
            return default_res


def chunkize_serial(iterable, chunksize, as_numpy=False):
    """
    Return elements from the iterable in `chunksize`-ed lists. The last returned
    element may be smaller (if length of collection is not divisible by `chunksize`).

    >>> print(list(grouper(range(10), 3)))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    """
    import numpy
    it = iter(iterable)
    while True:
        if as_numpy:
            # convert each document to a 2d numpy array (~6x faster when transmitting
            # chunk data over the wire, in Pyro)
            wrapped_chunk = [[numpy.array(doc) for doc in itertools.islice(it, int(chunksize))]]
        else:
            wrapped_chunk = [list(itertools.islice(it, int(chunksize)))]
        if not wrapped_chunk[0]:
            break
        # memory opt: wrap the chunk and then pop(), to avoid leaving behind a dangling reference
        yield wrapped_chunk.pop()

grouper = chunkize_serial


class RepeatCorpusNTimes(SaveLoad):

    def __init__(self, corpus, n):
        """
        Repeat a `corpus` `n` times.

        >>> corpus = [[(1, 0.5)], []]
        >>> list(RepeatCorpusNTimes(corpus, 3)) # repeat 3 times
        [[(1, 0.5)], [], [(1, 0.5)], [], [(1, 0.5)], []]
        """
        self.corpus = corpus
        self.n = n

    def __iter__(self):
        for _ in xrange(self.n):
            for document in self.corpus:
                yield document