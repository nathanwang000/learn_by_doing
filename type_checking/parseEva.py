from rtn import RTN, EMPTY, handle_error, push, cast, addSem, pop, ERROR, replace
from tokenizer import Tokenizer, TokenSpec, Token
'''
example eva program

(begin (var x 10) (+ x 20))

So the syntax contains

Exp = Atom | List
Atom = NUMBER | STRING | SYMBOL
List = LPAREN ListEntries RPAREN
ListEntries = Exp ListEntries | EMPTY

# my RTN cannot handle left recursion but can handle right recursion correctly
'''

eva_token_spec = [
    ('', r'#.*'),     # Comment (ignored)
    ('', r'\s+'),     # Skip over spaces and tabs
    ('NUMBER',   r'\d+(\.\d*)?'),  # Integer or decimal number
    ('STRING',   r'"[^"\\]*"'),  # String literal
    # symbol: word or + - * / = < >
    ('SYMBOL',   r'[\w\+\-\*\/=<>]+'),    # Identifiers
    ('LPAREN',   r'\('),           # Left Parenthesis
    ('RPAREN',   r'\)'),           # Right Parenthesis
    # '[1, 2, 3]' some native list syntax
    ('LBRACKET', r'\['),           # Left Bracket
    ('RBRACKET', r'\]'),           # Right Bracket
    ]

# try running rtn to match empty string or 'a'
def token_rtn(token_name):

    @handle_error(f'token_rtn({token_name})')
    def _f(s, r):
        if s and s[0].type == token_name:
            r = r + [s[0].value]
            return s[1:], r

    return RTN(_f)
    

NUMBER = cast(float, token_rtn('NUMBER'))
STRING = token_rtn('STRING')
SYMBOL = token_rtn('SYMBOL')
LPAREN = pop(token_rtn('LPAREN'))
RPAREN = pop(token_rtn('RPAREN'))

Atom = NUMBER + STRING + SYMBOL

@RTN
def ListEntries(s, r):
    # machine = Exp * ListEntries + EMPTY
    machine = addSem(Exp * ListEntries, lambda r: r[:-2] + [[r[-2]] + r[-1]]) + \
        push([], EMPTY)
    return machine(s, r)

List = LPAREN * ListEntries * RPAREN
Exp = Atom + List

def parse(src: str, rtn: RTN = Exp):
    tokenizer = Tokenizer([TokenSpec(name, pattern) for name, pattern in eva_token_spec])
    tokens = list(tokenizer.tokenize(src))
    s, r = rtn(tokens, [])
    if (s, r) == ERROR or s != [] or len(r) == 0:
        raise SyntaxError('Parsing error')
    return r[-1]


def main():
    # example using NUMBER
    # print(Atom([Token('NUMBER', '10')], []))
    # tokens = list(tokenizer.tokenize('(a b c)'))

    src = '(begin (var x 10) (+ x 2)) # this is a comment\n'
    print('Source:', src)

    result = parse(src)
    print('Parsed result:', result)


    
    
if __name__ == '__main__':
    main()
    

