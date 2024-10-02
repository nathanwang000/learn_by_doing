"""
a general definition of RTN for parsing
"""

import re
import os
import glob
import operator
from functools import wraps, reduce
from collections import defaultdict
from typing import Set, Tuple, Callable, Any, NamedTuple

class RTN:
    """
    general parser

    input:
      s: str
      r: semantic stack

    output: set of the following (for multiple possible parses)
      s: str
      r: semantic stack
      n: next characters
      m: remaining RTN after parsing

    examples:
    C("abc")("ab")[2] => {c}
    C("abc")("a")[2] => {b}
    C("abc")()[2] => {a}

    C("abc")("a") => "", ["a"], C("bc"), 'b'
    """

    def __init__(self, f):
        self.f = handle_left_recursion(f)  # avoid infinite recursion
        self.__doc__ = f.__doc__
        self.__name__ = f.__name__

    def __call__(self, s, r):
        return self.f(s, r)

    def __add__(self, other):
        assert isinstance(other, RTN), other
        return coproduct(self, other)

    def __mul__(self, other):
        assert isinstance(other, RTN), other
        return product(self, other)

    def __or__(self, other):
        assert isinstance(other, RTN), other
        return orproduct(self, other)


class RTN_UNIT(NamedTuple):
    s: str # remaining s after parsing
    r: Tuple # semantic stack
    n: str # next completion

RTN_OUTPUT = Set[RTN_UNIT]


def handle_left_recursion(f: RTN) -> RTN:
    "if seen and not computable automatically fail"
    mem = {}
    seen = defaultdict(int)

    @wraps(f)
    def _f(s, r):
        args = tuple(s) + tuple(r)
        if args in mem:
            return mem[args]
        if seen[args] >= 2:
            return Set()  # fail fast
        seen[args] += 1
        res = f(s, r)
        mem[args] = res
        return res

    return _f


def run(rtn: RTN, s: str) -> RTN_OUTPUT:
    return rtn(s, tuple())


@RTN
def ID(s: str, r: Tuple) -> RTN_OUTPUT:
    return {RTN_UNIT(s, r, "")}


def C(a: str) -> RTN:
    """
    >>> a = run(C("a"), "abc")
    >>> len(a)
    1
    >>> a.pop()[:3]
    ('bc', ('a',), '')
    >>> run(C("a"), "bc") # fail to parse
    set()
    >>> run(C("dc"), 'dc').pop()[:3]
    ('', ('dc',), '')

    # partial compeltion
    >>> sorted(extract_completions(run(C("abc"), "a")))
    ['abc']
    """

    @RTN
    def _f(s, r):
        if len(s) == 0:
            return {RTN_UNIT(s, r, a)}
        if s.startswith(a):
            # completed a but may have remaining
            return {RTN_UNIT(s[len(a) :], r + (a,), "")}

        # cascading
        """
        text: str, the text to match
        commands: list of str, the vocabulary to match against

        cascade rules for matching, if fail on one level, fall back
        to next level, return results
        a) starts with the text
        b) regular expression
        c) contains the text
        d) contains the text (case insensitive)
        e) todo: semantic search
        """
        rules = [
            lambda x: x.startswith(s),
            lambda x: re.match(s, x),
            lambda x: s in x,
            lambda x: s.lower() in x.lower(),
        ]
        for i, rule in enumerate(rules):
            try:
                if rule(a):
                    # partial completion
                    return {RTN_UNIT(s, r, a)}
            except:
                continue

        # failed to parse: _f != ID and s != ""
        return set()

    return _f


def product(a: RTN, b: RTN) -> RTN:  # compose
    """
    >>> a = run(C("a") * C("b"), "ab")
    >>> len(a)
    1
    >>> a.pop()[:3]
    ('', ('a', 'b'), '')
    >>> a = run(C("a") * C("b"), "a")
    >>> len(a)
    1
    >>> a.pop()[:3]
    ('', ('a',), 'b')
    """

    @RTN
    def _f(s, r):
        results = set()
        srnm0s = a(s, r)
        if len(srnm0s) == 0:  # error
            return results

        for srnm in srnm0s:
            s, r, n = srnm
            if n == "":  # exhasted first rtn
                srnm1s = b(s, r)
                if len(srnm1s) == 0:  # error
                    continue
                for srnm1 in srnm1s:
                    results.add(srnm1)
            else:
                results.add(RTN_UNIT(s, r, n))
        return results

    return _f


def coproduct(a: RTN, b: RTN) -> RTN:  # add
    """
    >>> run(C("a") + C("b"), "ab").pop()[:3]
    ('b', ('a',), '')
    >>> run(C('a') + C('bc') + C('de'), "dc")
    set()
    >>> a = run(C('a') + C('b') + C('c'), '')
    >>> sorted(extract_completions(a))
    ['a', 'b', 'c']
    """

    @RTN
    def _f(s: str, r: Tuple) -> RTN_OUTPUT:
        results = set()
        srmn0s = a(s, r)
        if len(srmn0s) == 0:
            return b(s, r)
        for srmn0 in srmn0s:
            results.add(srmn0)

        srmn1s = b(s, r)
        if len(srmn1s) == 0:
            return results
        for srmn1 in srmn1s:
            results.add(srmn1)

        return results

    return _f


def orproduct(a: RTN, b: RTN) -> RTN:  # take first qualified
    """
    behaves like or in python: take the first success case

    >>> extract_semantic_stacks(run(C('a') | C('ab'), 'a'))
    {('a',)}
    """

    @RTN
    def _f(s: str, r: Tuple) -> RTN_OUTPUT:
        results = set()
        srmn0s = a(s, r)
        if len(srmn0s) == 0:
            return b(s, r)
        for srmn0 in srmn0s:
            results.add(srmn0)

        if results:
            return results

        srmn1s = b(s, r)
        if len(srmn1s) == 0:
            return results
        for srmn1 in srmn1s:
            results.add(srmn1)

        return results

    return _f


def extract_remaining_strings(outputs: RTN_OUTPUT) -> Set[str]:
    return {o[0] for o in outputs}


def extract_semantic_stacks(outputs: RTN_OUTPUT) -> Set[str]:
    return {o[1] for o in outputs}


def extract_completions(outputs: RTN_OUTPUT) -> Set[str]:
    return {o[2] for o in outputs}


def extract_remaining_rtns(outputs: RTN_OUTPUT) -> Set[str]:
    return {o[3] for o in outputs}


###### sem manipulation
def addSem(f: RTN, sem: Callable[Tuple, Tuple]) -> RTN:  # semantics
    @RTN
    def _f(s, r):
        srnms = f(s, r)
        if len(srnms) == 0:
            return srnms

        return {RTN_UNIT(s, sem(r), n) for s, r, n in srnms}

    return _f


def push(symbol, rtn: RTN) -> RTN:
    """
    >>> o = run(push(1, C('c')), 'c')
    >>> extract_semantic_stacks(o)
    {('c', 1)}
    """
    return addSem(rtn, lambda r: r + (symbol,))


def replace(symbol, rtn: RTN) -> RTN:
    """
    >>> o = run(replace(1, C('c')), 'c')
    >>> extract_semantic_stacks(o)
    {(1,)}
    """
    return addSem(rtn, lambda r: r[:-1] + (symbol,))


def cast(f: Callable, rtn: RTN) -> RTN:
    """
    >>> o = run(cast(int, C('0')), '0')
    >>> extract_semantic_stacks(o)
    {(0,)}
    """
    assert callable(f), f"{f} must be callable"
    return addSem(rtn, lambda r: r[:-1] + (f(r[-1]),))


def pop(rtn):
    """pop the semantic stack
    >>> o = run(pop(C('0')), '0')
    >>> extract_semantic_stacks(o)
    {()}
    """
    return addSem(rtn, lambda r: r[:-1])


#### common RTN shorthands
def C_regex(a: str):
    """a is an regex

    >>> o = run(C_regex('a.*d'), 'adcb')
    >>> o.pop()[:3]
    ('cb', ('ad',), '')
    """

    @RTN
    def _f(s, r):
        if len(s) == 0:
            return {RTN_UNIT(s, r, a, _f)}

        results = set()
        m = re.match(a, s)  # match always match from begining
        if m:
            _s, _e = m.span()
            assert _s == 0, "bug: should always starts at 0"
            return {RTN_UNIT(s[_e:], r + (s[_s:_e],), "")}

        return results

    return _f


def cast_rtn(x: Any) -> RTN:
    if isinstance(x, RTN):
        return x
    if isinstance(x, str):
        return C(x)
    raise Exception(f"not supported type for {x} of type {type(x)}")


def C_operator(f) -> RTN:
    """
    f is an operator like operator.add

    # coproduct of choices
    >>> o = run(C_operator(operator.add)(['0', '1', '2']), '102')
    >>> extract_remaining_strings(o)
    {'02'}

    # product of choices
    >>> o = run(C_operator(operator.mul)(['0', '1', '2']), '012')
    >>> extract_remaining_strings(o)
    {''}

    # or of choices
    >>> o = run(C_or(['0', '1', '2']), '012')
    >>> len(o)
    1
    """

    def _f(choices) -> RTN:
        return reduce(f, map(cast_rtn, choices))

    return _f


C_add = C_operator(operator.add)
C_mul = C_operator(operator.mul)
C_or = C_operator(operator.__or__)

DIGIT = reduce(operator.add, (replace(i, C(str(i))) for i in range(10)))


#### REGEX reimplementation
def re_star(rtn: RTN) -> RTN:
    """
    a*: ID | a | a a*
    >>> s = "0000111"
    >>> sorted(extract_completions(run(re_star(DIGIT), s)))
    ['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    """

    @RTN
    def _f(s, r):  # lazy recursion
        # this is a trick to get away with max recursion
        return (ID + rtn + rtn * _f)(s, r)

    return _f


def re_plus(rtn: RTN) -> RTN:
    """
    a+: a | a a*
    >>> s = "0000111"
    >>> sorted(extract_completions(run(re_plus(DIGIT), s)))
    ['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    """

    @RTN
    def _f(s, r):  # lazy recursion
        # this is a trick to get away with max recursion
        return (rtn + rtn * _f)(s, r)

    return _f


DIGITS = re_plus(DIGIT)


##### example Json parsing
def get_number_list_rtn() -> RTN:
    """
    list: [] | [DIGITS] | [DIGITS, REST]
    REST: DIGITS | DIGITS, REST
    >>> sorted(extract_completions(run(get_number_list_rtn(), "[")))
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ']']

    >>> sorted(extract_completions(run(C_mul(['ab', 'abc', 'def']), 'a')))
    ['ab']
    """

    @RTN
    def number_list(s, r) -> RTN_OUTPUT:
        @RTN
        def REST(s, r):
            return (DIGITS + DIGITS * C(",") * REST)(s, r)

        return (C_mul("[]") + C("[") * DIGITS * C("]") + REST)(s, r)

    return number_list


##### example config.options parsing
def get_diary_config_rtn() -> RTN:
    """
    {config}.{options}
    diary_config: '<config>' * '.' * '<options>'
    config: 'config1' | 'config2' | 'config3'
    options: 'options1' | 'options2' | 'options3'

    >>> sorted(extract_completions(\
        run(get_diary_config_rtn(), ""),\
    ))
    ['config1', 'config2', 'config3']

    >>> sorted(extract_completions(\
        run(get_diary_config_rtn(), "con"),\
    )) # partial complete always suggest the full completion
    ['config1', 'config2', 'config3']

    >>> sorted(extract_completions(\
        run(get_diary_config_rtn(), 'config2'),\
    ))
    ['.']

    >>> sorted(extract_completions(\
        run(get_diary_config_rtn(), 'config3.'),\
    ))
    ['options1', 'options2', 'options3']

    >>> sorted(extract_completions(\
        run(get_diary_config_rtn(), 'config3.option'),\
    ))
    ['options1', 'options2', 'options3']
    """

    @RTN
    def diary_config(s, r) -> RTN_OUTPUT:
        return (
            C_add(["config1", "config2", "config3"])
            * C(".")
            * C_add(["options1", "options2", "options3"])
        )(s, r)

    return diary_config


def get_path_rtn():
    """parse valid path

    >>> sorted(extract_completions(run(get_path_rtn(), "/usr/lo")))
    ['local/']

    >>> sorted(extract_completions(run(get_path_rtn(), "asdfeaesf")))
    []
    """

    @RTN
    def _f(s: str, r: Tuple) -> RTN_OUTPUT:
        results = set()
        for path in glob.glob(os.path.expanduser(s) + "*"):
            postfix = "/" if os.path.isdir(path) else ""
            if path == s:
                results.add(("", r + (path,), "", ID))
            else:
                results.add(
                    (
                        os.path.basename(s),
                        r + (os.path.dirname(s),),
                        os.path.basename(path) + postfix,
                        ID,  # dummy f
                    )
                )

        return results

    return _f
