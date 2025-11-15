import re
from collections import namedtuple

# tokenizer: lexical analyzer
Token = namedtuple('Token', ['type', 'value'])
TokenSpec = namedtuple('TokenSpec', ['name', 'pattern'])

class Tokenizer:

    def __init__(self, token_spec: [TokenSpec]):
        # mark empty spec name as __SKIP{uniq number}
        for i, spec in enumerate(token_spec):
            if spec.name == '':
                token_spec[i] = TokenSpec(f'__SKIP{i}', spec.pattern)
        
        self.token_regex = re.compile('|'.join(f'(?P<{spec.name}>{spec.pattern})' for spec in token_spec))

    def tokenize(self, src: str)->[Token]:
        '''
        Tokenize the input source string into a sequence of Tokens.

        >>> _test_eva_tokenizer()
        [Token(type='LPAREN', value='('), Token(type='SYMBOL', value='begin'), Token(type='LPAREN', value='('), Token(type='SYMBOL', value='var'), Token(type='LPAREN', value='('), Token(type='SYMBOL', value='x'), Token(type='SYMBOL', value='number'), Token(type='RPAREN', value=')'), Token(type='NUMBER', value='10'), Token(type='RPAREN', value=')'), Token(type='RPAREN', value=')'), Token(type='LPAREN', value='('), Token(type='SYMBOL', value='+'), Token(type='NUMBER', value='2'), Token(type='SYMBOL', value='x'), Token(type='RPAREN', value=')')]
        '''
        pos = 0
        while pos < len(src):
            m = self.token_regex.match(src, pos)
            if not m:
                raise SyntaxError(f'Unexpected character: {src[pos]} at position {pos}')
            typ = m.lastgroup
            pos = m.end()
            if not typ.startswith('__SKIP'):
                val = m.group(typ)
                yield Token(typ, val)

def _test_eva_tokenizer():
    eva_token_spec = [
        ('', r'#.*'),     # Comment (ignored)
        ('', r'\s+'),     # Skip over spaces and tabs
        ('NUMBER',   r'\d+(\.\d*)?'),  # Integer or decimal number
        ('STRING',   r'"[^"\\]*"'),  # String literal
        # symbol: word or + - * / = < >
        ('SYMBOL',   r'[\w\+\-\*\/=<>]+'),    # Identifiers
        ('LPAREN',   r'\('),           # Left Parenthesis
        ('RPAREN',   r'\)'),           # Right Parenthesis
        ]
    eva_token_spec = [TokenSpec(name, pattern) for name, pattern in eva_token_spec]

    tokenizer = Tokenizer(eva_token_spec)
    src = '(begin (var (x number) 10)) # this is a comment\n(+ 2 x)'
    print(list(tokenizer.tokenize(src)))


if __name__ == '__main__':
    _test_eva_tokenizer()
