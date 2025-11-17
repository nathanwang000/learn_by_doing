'''
example eva program

(begin (var x 10) (+ x 20))

my RTN cannot handle left recursion but can handle right recursion correctly
(all left recursion can be transformed to right recursion)
'''

from rtn import EMPTY, RTN, addSem, cast, handle_error, pop, push
from tokenizer import Tokenizer, TokenSpec


def token_rtn(token_name):

    @handle_error(f'token_rtn({token_name})')
    def _f(s, r):
        if s and s[0].type == token_name:
            r = r + [s[0].value]
            return s[1:], r

    return RTN(_f)


class Parser:

    @classmethod
    def get_token_spec(cls) -> list[tuple[str, str]]:
        raise NotImplementedError(
            'Subclasses must implement get_token_spec method')

    @classmethod
    def get_grammar_rtn(cls, rtn_name: str) -> RTN:
        raise NotImplementedError(
            'Subclasses must implement get_grammar_rtn method')

    @classmethod
    def parse(cls, src: str, rtn_name: str | None = None):
        rtn = cls.get_grammar_rtn(rtn_name)

        tokenizer = Tokenizer([
            TokenSpec(name, pattern) for name, pattern in cls.get_token_spec()
        ])
        tokens = list(tokenizer.tokenize(src))
        s, r = rtn(tokens, [])
        if s is None or s != [] or len(r) == 0:
            raise SyntaxError(
                f'Parsing error in parsing {[_c.value for _c in tokens]}:\nremaining tokens = {[_c.value for _c in s] if s else s}\nsemantc stack = {r}'
            )
        return r.pop()


class EvaParser(Parser):
    '''
    example eva program

    (begin (var x 10) (+ x 20))

    So the syntax contains

    Exp = Atom | List
    Atom = NUMBER | STRING | SYMBOL
    List = LPAREN ListEntries RPAREN
    ListEntries = Exp ListEntries | EMPTY

    >>> EvaParser.parse('(begin (var x 10) (+ x 2))')
    ['begin', ['var', 'x', 10.0], ['+', 'x', 2.0]]
    '''

    @classmethod
    def get_token_spec(cls) -> list[tuple[str, str]]:
        return [
            ('', r'#.*'),  # Comment (ignored)
            ('', r'\s+'),  # Skip over spaces and tabs
            ('NUMBER', r'\d+(\.\d*)?'),  # Integer or decimal number
            ('STRING', r'"[^"\\]*"'),  # String literal
            # symbol: word or + - * / = < >
            ('SYMBOL', r'[\w\+\-\*\/=<>!,\.]+'),  # Identifiers
            ('LPAREN', r'\('),  # Left Parenthesis
            ('RPAREN', r'\)'),  # Right Parenthesis
            # '[1, 2, 3]' some native list syntax
            ('LBRACKET', r'\['),  # Left Bracket
            ('RBRACKET', r'\]'),  # Right Bracket
        ]

    @classmethod
    def get_grammar_rtn(cls, rtn_name: str | None = None) -> RTN:
        NUMBER = cast(float, token_rtn('NUMBER'))
        STRING = token_rtn('STRING')
        SYMBOL = token_rtn('SYMBOL')
        LPAREN = pop(token_rtn('LPAREN'))
        RPAREN = pop(token_rtn('RPAREN'))

        Atom = NUMBER + STRING + SYMBOL

        @RTN
        def ListEntries(s, r):
            # machine = Exp * ListEntries + EMPTY
            machine = addSem(Exp * ListEntries,
                lambda r: r[:-2] + [[r[-2]] + r[-1]]) + \
                push([], EMPTY)
            return machine(s, r)

        List = LPAREN * ListEntries * RPAREN
        Exp = Atom + List

        if not rtn_name:
            return Exp

        if rtn_name in locals():
            ret = locals()[rtn_name]
            assert isinstance(ret, RTN), f'{rtn_name} is not of type RTN'
            return ret

        raise ValueError(f'No such rtn: {rtn_name}')


class EvaFunctionStringParser(Parser):
    '''
    A parser for eva function from string

    Fn[(number, number) -> number]
    The grammar is as follows:

        ATOM := 'number' | 'string' | 'boolean'
        FType := 'Fn[' '(' Types ')' '->' Type ']'
        Types := Type ',' Types | Type | ε
        Type := ATOM | FType


    >>> EvaFunctionStringParser.parse('Fn[(number, string) -> boolean]')
    {'name': 'Fn', 'args': ['number', 'string'], 'return_type': 'boolean'}

    # nested function type
    >>> EvaFunctionStringParser.parse('Fn[(number, Fn[(string) -> boolean]) -> string]')
    {'name': 'Fn', 'args': ['number', {'name': 'Fn', 'args': ['string'], 'return_type': 'boolean'}], 'return_type': 'string'}

    # function with no arguments
    >>> EvaFunctionStringParser.parse('Fn[() -> number]')
    {'name': 'Fn', 'args': [], 'return_type': 'number'}
    '''

    @classmethod
    def get_token_spec(cls) -> list[tuple[str, str]]:
        return [
            ('ARROW', r'->'),
            ('COMMA', r','),
            ('LPAREN', r'\('),
            ('RPAREN', r'\)'),
            ('LBRACKET', r'\['),
            ('RBRACKET', r'\]'),
            ('ATOM', r'number|string|boolean'),
            ('FN', r'Fn'),
            ('', r'\s+'),  # skip spaces
        ]

    @classmethod
    def get_grammar_rtn(cls, rtn_name: str | None = None) -> RTN:
        ATOM = token_rtn('ATOM')
        FN = pop(token_rtn('FN'))
        ARROW = pop(token_rtn('ARROW'))
        COMMA = pop(token_rtn('COMMA'))
        LPAREN = pop(token_rtn('LPAREN'))
        RPAREN = pop(token_rtn('RPAREN'))
        LBRACKET = pop(token_rtn('LBRACKET'))
        RBRACKET = pop(token_rtn('RBRACKET'))

        @RTN
        def FType(s, r):
            machine = FN * LBRACKET * LPAREN * Types * RPAREN * ARROW * Type * RBRACKET
            machine = addSem(
                machine, lambda r: r[:-2] + [{
                    'name': 'Fn',
                    'args': r[-2],
                    'return_type': r[-1]
                }])
            return machine(s, r)

        @RTN
        def Types(s, r):
            machine = addSem(Type * COMMA * Types,
                lambda r: r[:-2] + [[r[-2]] + r[-1]]) + \
                cast(lambda x: [x], Type) + push([], EMPTY)
            return machine(s, r)

        Type = ATOM + FType

        if not rtn_name:
            return FType

        if rtn_name in locals():
            ret = locals()[rtn_name]
            assert isinstance(ret, RTN), f'{rtn_name} is not of type RTN'
            return ret

        raise ValueError(f'No such rtn: {rtn_name}')


def main():
    # example using NUMBER
    # print(Atom([Token('NUMBER', '10')], []))
    # tokens = list(tokenizer.tokenize('(a b c)'))

    src = '(begin (var x 10) (+ x 2)) # this is a comment\n'
    print('Source:', src)

    result = EvaParser.parse(src)
    print('Parsed result:', result)


if __name__ == '__main__':
    main()
