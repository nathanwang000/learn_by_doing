"""
a general definition of RTN for parsing

Limitations: it won't greedily match, i.e., it always tries the first option in coproduct first. This means that if two options can match the same prefix, the second option will never be tried.

e.g., C('a') + C('ab') will never match 'ab' since C('a') will always succeed first.

This however means that the runtime is linear in the length of the input string for non-left-recursive grammars.

Maybe in future version we can add backtracking to handle such cases. Like using generators to yield all possible matches (unlike now a single match) and let the caller decide which one to use.
"""

import re

from functools import wraps
from collections import defaultdict
from collections.abc import Iterable
from typing import Callable

VERBOSE = False

class Node:
    def __init__(self, val, args=[]):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    def __repr__(self):
        return f"Node({self.val}, {self.args})"


class RTN:
    """
    general parser

    input:
      s: str
      r: semantic stack

    output:
      s: str
      r: semantic stack
    """

    def __init__(self, f):
        self.f = handle_left_recursion(f)  # avoid infinite recursion
        self.__doc__ = f.__doc__
        self.__name__ = f.__name__

    def __call__(self, s, r):
        return self.f(s, r)

    def __add__(self, other):
        assert type(other) is RTN, other
        return coproduct(self, other)

    def __mul__(self, other):
        assert type(other) is RTN, other
        return product(self, other)


def to_rtn(f):  # make function result an RTN
    @wraps(f)
    def _f(*args, **kwargs):
        return RTN(f(*args, **kwargs))

    return _f


ERROR = (None, None)


def is_error(e):
    return e[0] is None or e[1] is None


def recursive_tuple(iterable):
    # convert iterable to nested tuple
    # frst check it is iterable, if not just return as is
    if isinstance(iterable, Iterable) and not isinstance(iterable, (str, bytes)):
        return tuple(recursive_tuple(i) for i in iterable)
    return iterable

def handle_left_recursion(f):
    "if seen and not computable automatically fail"
    mem = {}
    seen = defaultdict(int)

    @wraps(f)
    def _f(s, r):
        args = (recursive_tuple(s), recursive_tuple(r))
        if args in mem:
            return mem[args]
        if seen[args] >= 2:
            if VERBOSE:
                print(f"left recursion detected in {f} with s={s}, r={r}")
            return ERROR
        seen[args] += 1
        res = f(s, r)
        mem[args] = res
        return res

    return _f

def handle_error(note=None):
    """decorator to handle errors in RTN functions"""
    def handler(f):
        @wraps(f)
        def _f(s, r):
            try:
                s, r = f(s, r)
                return s, r
            except:
                error_note = f"`{note}`" if note else f"{f}"
                if VERBOSE:
                    print(f"error in {error_note} with s={s}, r={r}")
                return ERROR

        return _f
    return handler


@RTN
def ID(s, r):
    return s, r


@to_rtn
def product(a, b):  # compose
    def _f(s, r):
        sr = a(s, r)
        if is_error(sr):
            return ERROR
        s, r = sr
        return b(s, r)

    return _f


@to_rtn
def coproduct(a, b):  # or
    def _f(s, r):
        sr = a(s, r)
        if not is_error(sr):
            return sr
        return b(s, r)

    return _f


@to_rtn
def addSem(f: RTN, sem: Callable):  # semantics
    """sem is a function to modify the semantic stack"""

    assert callable(sem), f"{sem} must be callable"
    assert isinstance(f, RTN), f'f must be of type RTN but got {type(f)}'
    
    @handle_error(f'addSem({f}, {sem})')
    def _f(s, r):
        s, r = f(s, r)
        return s, sem(r)

    return _f


@to_rtn
def consumer(a):
    """a is a condition"""

    @handle_error('consumer(condition)')
    def _f(s, r):
        if a(s[0]):
            r = r + [s[0]]
            return s[1:], r

    return _f


@to_rtn
def C_regex(a):
    """a is an regex"""

    @handle_error(f'C_regex(`{a}`)')
    def _f(s, r):
        m = re.match(a, s[0])
        if m:
            r = r + [m.group()]
            return s[1:], r

    return _f


def C(x)->RTN:
    # return consumer(lambda a: re.match("^" + x + "$", a))
    # the later is more interpretable in error messages
    return C_regex("^" + x + "$")

EMPTY = RTN(lambda s, r: (s, r))  # match empty string

# auxiliary semantic stack operations
def push(symbol, rtn)->RTN:
    assert isinstance(rtn, RTN), f'rtn must be of type RTN but got {type(rtn)}'
    return addSem(rtn, lambda r: r + [symbol])

def replace(symbol, rtn)->RTN:
    assert isinstance(rtn, RTN), f'rtn must be of type RTN but got {type(rtn)}'    
    return addSem(rtn, lambda r: r[:-1] + [symbol])

def cast(f, rtn)->RTN:
    '''f must be callable'''
    assert callable(f), f"{f} must be callable"
    assert isinstance(rtn, RTN), f'rtn must be of type RTN but got {type(rtn)}'    
    return addSem(rtn, lambda r: r[:-1] + [f(r[-1])])

def pop(rtn)->RTN:
    '''pop the semantic stack'''
    assert isinstance(rtn, RTN), f'rtn must be of type RTN but got {type(rtn)}'    
    return addSem(rtn, lambda r: r[:-1])

def Star(rtn)->RTN:
    '''
    zero or more repetitions of rtn

    >>> Star(C('a'))(list('aaab'), [])
    (['b'], ['a', 'a', 'a'])
    '''
    assert isinstance(rtn, RTN), f'rtn must be of type RTN but got {type(rtn)}'    
    @RTN
    def _f(s, r):
        machine = rtn * _f + EMPTY
        return machine(s, r)
    return _f
