'''
smoosh: a funny language created by Jx because he likes being smooshed.
  I kind of want to call it coq, but that name is already taken.
  This is a joke language with lisp syntax and allow simple proof like lean.
'''
from typing import Tuple

def tokenize(code: str) -> list[str]:
    '''
    Tokenize the input code into a list of tokens
    code: str - the input code as a string

    >>> code = "(define (square  x) (* x x))"
    >>> tokenize(code)
    ['(', 'define', '(', 'square', 'x', ')', '(', '*', 'x', 'x', ')', ')']
    '''
    return code.replace('(', ' ( ').replace(')', ' ) ').split()

def parse(tokens: list[str]) -> str | list[str]:
    '''
    Parse the list of tokens into an abstract syntax tree (AST)
    tokens: list[str] - the list of tokens
    >>> tokens = tokenize("(define (square  x) (* x x))")
    >>> len(tokens)
    12
    >>> parse(tokens)
    ['define', ['square', 'x'], ['*', 'x', 'x']]
    '''
    
    def parse_helper(tokens: list[str], token_pos=0) -> Tuple[str|list[str], int]:
        '''
        Parse the list of tokens into an abstract syntax tree (AST)
        The result will be ["operator" operand1 operand2 ...] or a single token

        tokens: list[str] - the list of tokens

        >>> tokens = tokenize("(define (square  x) (* x x))")
        >>> parse_helper(tokens)
        (['define', ['square', 'x'], ['*', 'x', 'x']], 12)
        '''
        if token_pos >= len(tokens):
            raise SyntaxError("Unexpected EOF while reading")

        token = tokens[token_pos]
        if token == '(':
            token_pos += 1
            lst = []
            while tokens[token_pos] != ')':
                element, token_pos = parse_helper(tokens, token_pos)
                lst.append(element)
            token_pos += 1
            return lst, token_pos
        elif token == ')':
            raise SyntaxError("Unexpected )")
        else:
            token_pos += 1
            return token, token_pos

    ast, token_pos = parse_helper(tokens, 0)
    if token_pos != len(tokens):
        raise SyntaxError(f"Unexpected tokens after parsing: {tokens[token_pos:]}")
    return ast

class Type:
    '''
    A simple representation of types in smoosh language.
    '''
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f'Type({self.name})'

    @classmethod
    def function_type(cls, arg_type: 'Type', return_type: 'Type') -> 'Type':
        return cls(f'({arg_type} -> {return_type})')

    @classmethod
    def inductive_type(cls, name: str, constructors: dict[str, 'Type']) -> 'Type':
        return cls(f'Inductive {name} with constructors {constructors}')

    @classmethod
    def pi_type(cls, var_name: str, var_type: 'Type', return_type: 'Type') -> 'Type':
        return cls(f'Pi({var_name}: {var_type}) -> {return_type}')

    @classmethod
    def constraint_type(cls, base_type: 'Type', constraint: str) -> 'Type':
        return cls(f'Constraint({constraint}) on {base_type}')

    @classmethod
    def _compute_from_type(cls, type_obj: 'Type') -> 'Type':
        '''
        Create a Type from another Type object.
        For example, if the contraint type can be applied, it will try to
        evaluate the constraint and return the resulting type.
        '''
        pass

    @classmethod
    def _compute_from_string(cls, type_str: str) -> 'Type':
        '''
        Create a Type from a string representation.
        This is a placeholder implementation and should be expanded to handle
        all the type constructs in smoosh language.
        '''
        tokens = tokenize(type_str)
        return cls._compute_from_tokens(tokens)
    
    @classmethod
    def _compute_from_tokens(cls, tokens: list[str]) -> 'Type':
        '''
        Create a Type from a list of tokens.
        This is a placeholder implementation and should be expanded to handle
        all the type constructs in smoosh language.
        '''
        if tokens[0] == '->':
            arg_type = cls.from_tokens(tokens[1:2])
            return_type = cls.from_tokens(tokens[2:3])
            return cls.function_type(arg_type, return_type)
        elif tokens[0] == 'inductive':
            name = tokens[1]
            constructors = {}
            for i in range(2, len(tokens), 2):
                ctor_name = tokens[i]
                ctor_type = cls.from_tokens(tokens[i+1:i+2])
                constructors[ctor_name] = ctor_type
            return cls.inductive_type(name, constructors)
        # Add more parsing logic for constraints and pi types as needed
        else:
            return cls(tokens[0])

    
class SmooshTC:

    '''
    A simple type checker for the smoosh language.
    It is a dependently typed language like lean.

    example programs looks like

    (inductive claim (-> str Type)
        (mkclaim (-> (s: str) (claim s)))
        )
      
    (inductive evidence (-> str Type)
        (mkevidence (-> (s: str) (evidence s)))
        )

    # it should also support constraints on types, like only allowing certain types
    # to be created if certain conditions are met.
    (inductive positive_nat (-> Type)
        (mkposnat (-> (n: nat) (constraint (> 0 n) (positive_nat))))
        )

    The way to type check the above roughly follows:
    1. Parse the code into an AST
    2. Build a context of types and constructors (save them in a dictionary of variable to type; for closure, also point to parent context)
    3. For each expression, check its type according to the context (like for function types, just make sure the argument types match and return type matches; for inductive types, make sure the constructors are valid; for constraints, make sure the constraints are satisfied and then the resulting type is valid; for general Pi types, that is for all x : A, B(x), make sure A is a type and B(x) is a type for all x in A)
    '''
