import re
import parseEva

class Type:
    '''
    represents a type in eva language 
    '''
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Type) and self.name == other.name

    def __repr__(self):
        return self.name

    @classmethod
    def fromString(cls, typeString: str):
        '''
        return a Type that is registered

        >>> Type.fromString('number')
        number
        '''
        if hasattr(cls, typeString):
            return getattr(cls, typeString)

        raise Exception(f'Unknown type: "{typeString}"')


# built-in types
Type.number = Type('number')
Type.string = Type('string')
Type.boolean = Type('boolean')

class TypeEnvironment:
    '''
    type environment for eva language
    It maintains an env mapping of variable names to their types
    and a reference to a parent environment for nested scopes.
    ''' 
    def __init__(self, env, parent=None):
        self.parent = parent
        self.env = env

    def define(self, var_name: str, var_type: Type) -> Type:
        self.env[var_name] = var_type
        return var_type

    def lookup(self, varname:str) -> Type:
        if varname in self.env:
            return self.env[varname]
        elif self.parent is not None:
            return self.parent.lookup(varname)
        else:
            raise Exception(f'Undefined variable: "{varname}"')
        
    
class EvaTC:
    '''
    static type checker for eva language
    '''
    def __init__(self):
        self.global_env = TypeEnvironment({
            'VERSION': Type.string,
        })
        
    def _isNumber(self, exp):
        return isinstance(exp, int) or isinstance(exp, float)

    def _isString(self, exp):
        return isinstance(exp, str) and (exp.startswith('"') and exp.endswith('"'))

    def _isBoolean(self, exp):
        return exp in ['true', 'false']

    def tc(self, exp, env: TypeEnvironment=None):
        '''type checks an expression

        Args:
            exp: expression to be type checked

        >>> eva = EvaTC()
        >>> eva.tc(42)
        number
        >>> eva.tc('"hello"')
        string
        >>> eva.tc(['var', ['x', 'number'], 10])
        number
        >>> eva.tc("x")
        number
        >>> eva.tc(['set', 'x', 20])
        number
        >>> eva.tc('true')
        boolean
        '''
        if env is None: env = self.global_env
        
        if self._isNumber(exp):
            return Type.number
        
        if self._isString(exp):
            return Type.string
        
        if self._isBoolean(exp):
            return Type.boolean

        # math operations
        if self._isBinaryOp(exp):
            return self._binary(exp, env)

        # variable declaration e.g., (var x, 10)
        # with type check (var (x number) 'foo') # should raise type error
        if self._isOperand('var', exp):
            self._checkArity(exp, 2)
            var_name = exp[1]
            var_value = exp[2]
            # with type check: (var (x number) 'foo')
            if isinstance(var_name, list):
                self._checkArity(var_name, 1)
                actual_var_name = var_name[0]
                expected_var_type = Type.fromString(var_name[1])
                var_type = self.tc(var_value, env)
                self._expect(var_type, expected_var_type, var_value, exp)
                return env.define(actual_var_name, var_type)
            
            var_type = self.tc(var_value, env)
            return env.define(var_name, var_type)

        # variable access
        if self._isVariableName(exp):
            return env.lookup(exp)

        # (set x 10)
        if self._isOperand('set', exp):
            self._checkArity(exp, 2)
            var_name = exp[1]
            var_value = exp[2]
            var_type = self.tc(var_name, env)
            value_type = self.tc(var_value, env)
            return self._expect(value_type, var_type, var_value, exp)
        
        # block: sequence of expressions
        # (begin (var x 10) (var y 20) (+ x y))
        if self._isOperand('begin', exp):
            block_env = TypeEnvironment({}, parent=env)            
            return self._tcBlock(exp, block_env)

        # if expression: branches must have same type to type check w/o running
        if self._isOperand('if', exp):
            self._checkArity(exp, 3)
            cond_type = self.tc(exp[1], env)
            self._expect(cond_type, Type.boolean, exp[1], exp)
            then_type = self.tc(exp[2], env)
            else_type = self.tc(exp[3], env)
            return self._expect(then_type, else_type, exp, exp)

        # comparison operators: <, >, <=, >=, ==, !=
        if self._isOperand(['<', '>', '<=', '>=', '==', '!='], exp):
            self._checkArity(exp, 2)
            t1 = self.tc(exp[1], env)
            t2 = self.tc(exp[2], env)
            self._expect(t2, t1, exp[2], exp)
            return Type.boolean
        
        raise Exception(f'Unknown expression type: "{exp}"')

    def _tcBlock(self, exp, env):
        result_type = None
        for sub_exp in exp[1:]:
            result_type = self.tc(sub_exp, env)
            # debug
            # print(sub_exp, '=>', result_type)
            # print('env', env, 'parent', env.parent.env if env.parent else None)
            
        return result_type

    def _isOperand(self, op_name: str | list[str], exp):
        if isinstance(op_name, str):
            return isinstance(exp, list) and len(exp) > 0 and exp[0] == op_name
        elif isinstance(op_name, list):
            return isinstance(exp, list) and len(exp) > 0 and exp[0] in op_name
        return False
    
    def _isVariableName(self, exp)->bool:
        return isinstance(exp, str) and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', exp)
        
    def _binary(self, exp, env)->Type:
        self._checkArity(exp, 2)
        t1 = self.tc(exp[1], env)
        t2 = self.tc(exp[2], env)

        allowed_types = self._getAllowedTypes(exp[0])
        self._expctOperatorType(t1, allowed_types, exp)
        self._expctOperatorType(t2, allowed_types, exp)
        
        return self._expect(t2, t1, exp[2], exp)

    def _expctOperatorType(self, actualType: Type, allowedTypes: list[Type], exp):
        if actualType not in allowedTypes:
            raise Exception(f'Type error: expected one of "{allowedTypes}", got "{actualType}" in expression "{exp}"')
        
    def _getAllowedTypes(self, op)->list[Type]:
        if op in ['+', '-', '*', '/']:
            return {
                '+': [Type.number, Type.string],
                '-': [Type.number],
                '*': [Type.number],
                '/': [Type.number],
                }[op]
        raise Exception(f'Unknown operator: "{op}"')
        
    def _expect(self, actualType, expectedType, value, exp)->Type:
        if not actualType == expectedType:
            self._throw(actualType, expectedType, value, exp)
        return actualType

    def _throw(self, actualType, expectedType, value, exp):
        raise Exception(f'Type error: expected "{expectedType}", got "{actualType}" for value "{value}" in expression "{exp}"')
    
    def _isBinaryOp(self, exp)->bool:
        return isinstance(exp, list) and exp[0] in ['+', '-', '*', '/']

    def _checkArity(self, exp, expected_arity):
        if len(exp) - 1 != expected_arity:
            raise Exception(f'Arity error: expected {expected_arity}, got {len(exp) - 1} in expression {exp}')
        
        
def exec(eva, exp):
    if isinstance(exp, str):
        # add (begin ...) to make it a block so that globally I can just write
        # a sequence of expressions
        exp = parseEva.parse(f'(begin {exp})')

        # but now all sequence of expressions should be executed a global scope
        return eva._tcBlock(exp, eva.global_env)

    return eva.tc(exp)

def test(eva, exp, expected_type):
    inferred_type = exec(eva, exp)
    assert inferred_type == expected_type, f'Expected {expected_type}, but got {inferred_type} for expression {exp}'

if __name__ == '__main__':
    eva = EvaTC()

    # primitive types
    test(eva, 42, Type.number)
    test(eva, 3.14, Type.number)
    test(eva, '"hello"', Type.string)
    test(eva, '"world"', Type.string)

    # math 
    test(eva, ['+', 2, 3], Type.number)
    test(eva, ['/', 3, 0], Type.number)

    # string concatenation
    test(eva, ['+', '"foo"', '"bar"'], Type.string)
    # test(eva, ['-', '"foo"', '"bar"'], Type.string)    

    # variable declaration
    test(eva, ['var', 'x', 10], Type.number)

    # variable declaration with a type
    test(eva, ['var', ['y', 'number'], 'x'], eva.tc('x'))

    # variable access
    test(eva, 'x', Type.number)

    # global variable
    test(eva, 'VERSION', Type.string)

    # block: sequence of expressions
    test(eva,
         ['begin',
          ['var', 'x', 10],
          ['var', 'y', 20],
          ['+', 'x', 'y']
          ], Type.number)

    # block: local variables
    test(eva,
         ['begin',
          ['var', 'x', 10],
          ['var', 'y', 20],
          ['begin',
           ['var', 'x', '"hello"'],
           ['+', 'x', '" world"']
          ],
          ['+', 'x', 'y']
          ], Type.number)
    
    # block: accessing outer scope variable
    test(eva,
         ['begin',
          ['var', 'x', 10],
          ['begin',
           ['+', 'x', 5]
          ]
          ], Type.number)

    # parsing
    test(eva, '(var x 10) (var y 20)', Type.number)
    test(eva, '(begin (var x 10) (+ x 20))', Type.number)
    # test(eva, '(var x 10) (set x "hello")', Type.string)

    # control flow
    test(eva,
         '''
         (var x 10)
         (var y 20)
         (if (< x 10)
             (set y 1)
             (set y 2))
         y
         ''',
            Type.number
         )

    # print(parseEva.parse('(begin (var x 10) (var y 20))'))

    # introduce a list expression
    # print(parseEva.parse('(var x (list 1 2 3))')) # ['var', 'x', ['list', 1, 2, 3]]
    
    print("All tests passed.")



