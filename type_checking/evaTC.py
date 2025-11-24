import re

import parser  # pylint: disable=W4901 deprecated-module
import union_find

AliasUnionFind = union_find.UnionFind()

class Type:
    '''
    represents a type in eva language
    '''

    def __init__(self, name: str):
        '''
        initialize a Type instance
        '''
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Type):
            return False

        if AliasUnionFind.find(self.name) == AliasUnionFind.find(other.name):
            return True

        # inheritance
        if isinstance(other, ClassType) and isinstance(self, ClassType):
            return other == self.superclass

        return False

    def __repr__(self):
        return self.name

    @classmethod
    def fromString(cls, typeString: str | list):
        '''
        return a Type that is registered

        >>> Type.fromString('number')
        number
        '''
        if isinstance(typeString, list):
            # parse the list as function type
            return FunctionType.fromString(typeString)
        elif isinstance(typeString, str):
            if hasattr(cls, typeString):
                return getattr(cls, typeString)
            if typeString.startswith('(Fn'):
                # delegate to FunctionType fromString parser
                return FunctionType.fromString(typeString)

        raise Exception(f'Unknown type: "{typeString}"')


class FunctionType(Type):
    '''
    represents a function type in eva language
    e.g., (number, number) -> number

    >>> FunctionType([Type.number, Type.number], Type.number)
    (Fn (number number) number)
    '''

    def __init__(self, param_types: list[Type], return_type: Type):
        self.param_types = param_types
        self.return_type = return_type
        super().__init__(name=str(self))

    def __eq__(self, other):
        return isinstance(other, FunctionType) and self.name == other.name

    def __repr__(self):
        param_types_str = ' '.join([str(t) for t in self.param_types])
        return f'(Fn ({param_types_str}) {self.return_type})'

    @classmethod
    def fromString(cls, typeString: str | list):
        '''
        parse a function type string and return a FunctionType instance

        >>> FunctionType.fromString('(Fn (number number) number)')
        (Fn (number number) number)


        >>> FunctionType.fromString('(Fn (number) (Fn (number) number))')
        (Fn (number) (Fn (number) number))
        '''
        if isinstance(typeString, str):
            exp = parser.EvaParser.parse(typeString)
        elif isinstance(typeString, list):
            exp = typeString
        else:
            raise Exception(f'Invalid type string: "{typeString}"')

        assert exp[0] == 'Fn', f'Invalid function type string: "{typeString}"'
        param_types = [Type.fromString(t) for t in exp[1]]
        return_type = Type.fromString(exp[2])
        return cls(param_types, return_type)


class AliasType(Type):
    '''
    represents a type alias in eva language
    e.g., (type int number) # create an alias for the number type
    '''

    def __init__(self, alias_name: str, actual_type: Type):
        super().__init__(name=alias_name)
        self.actual_type = actual_type
        AliasUnionFind.union(alias_name, actual_type.name)

class ClassType(Type):
    '''
    represents a class type in eva language. It has properties and methods
    and implement inheritance.

    The underlying implementation is similar to TypeEnvironment

      (class <Name> <SuperclassName | null> <body>)

    e.g.
      (class Person null (begin
          (constructor ((self Person) (name string) (age number)) -> Person
          (begin
            (set (prop self name) name)
            (set (prop self age) age)
            self
          )
      )

    >>> eva = EvaTC()
    >>> eva.tc(parser.EvaParser.parse('(class Person null (begin))'))
    Person
    >>> eva.tc(parser.EvaParser.parse('(class Alice Person (begin))'))
    Alice
    >>> eva.tc('Alice') == eva.tc('Person')
    True
    >>> eva.tc('Person') == eva.tc('Alice')
    False
    '''
    def __init__(self, name: str, superclass: 'ClassType|Type.null'=Type('null')):
        super().__init__(name=name)
        self.superclass = superclass
        # env host all the methods and properties of the class
        # kind of like __dict__ in Python class
        self.env = TypeEnvironment({},
            parent=superclass.env if superclass != Type.null else None)

    def getField(self, name: str):
        return self.env.lookup(name)


# built-in types
Type.number = Type('number')
Type.string = Type('string')
Type.boolean = Type('boolean')
Type.null = Type('null')
Type.Function = FunctionType
Type.Alias = AliasType
Type.Class = ClassType


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

    def lookup(self, varname: str) -> Type:
        if varname in self.env:
            return self.env[varname]
        elif self.parent is not None:
            return self.parent.lookup(varname)
        else:
            raise Exception(f'Undefined variable: "{varname}"')

    def __repr__(self):
        return f'TypeEnvironment({self.env})'


class EvaTC:
    '''
    static type checker for eva language
    '''

    def __init__(self):
        self.global_env = TypeEnvironment({
            'VERSION':
            Type.string,
            'math.pi':
            Type.number,
            'math.sin':
            Type.fromString('(Fn (number) number)'),
        })

    def __repr__(self):
        return f'EvaTC(global_env={self.global_env})'

    def _isNumber(self, exp):
        return isinstance(exp, int) or isinstance(exp, float)

    def _isString(self, exp):
        return isinstance(exp, str) and (exp.startswith('"')
                                         and exp.endswith('"'))

    def _isBoolean(self, exp):
        return exp in ['true', 'false']

    def tc(self, exp, env: TypeEnvironment = None):
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

        # variable declaration e.g., (var x 10)
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

        # alias definition: e.g., (type int number)
        if self._isOperand('type', exp):
            self._checkArity(exp, 2)
            alias_name, actual_type_str = exp[1:]
            # assert alias is not already defined
            assert not hasattr(
                Type,
                alias_name), f'Type alias "{alias_name}" is already defined.'
            actual_type = Type.fromString(actual_type_str)
            setattr(Type, alias_name, AliasType(alias_name, actual_type))
            return getattr(Type, alias_name)

        # class definition: (class <Name> <SuperClass|null> <body>)
        if self._isOperand('class', exp):
            self._checkArity(exp, 3)
            name, superClassName, body = exp[1:]
            # resolve superclass
            superclass = getattr(Type, superClassName)
            assert superclass == Type.null or isinstance(superclass, Type.Class),\
        "Super class need to be Type.null or Type.Class"
            # define new class
            class_type = env.define(name, Type.Class(name, superclass))
            # let class assessible by name
            setattr(Type, name, class_type)
            self._tcBlock(body, class_type.env)
            return class_type

        # instance from a class
        # e.g. (new Person arg1 arg2)
        if self._isOperand('new', exp):
            name = exp[1]
            args = exp[2:]
            arg_types = [self.tc(arg, env) for arg in args]
            class_type = env.lookup(name)
            # should match constructor of the class
            constructor = class_type.getField('constructor')
            arg_types = [self.tc(arg, env) for arg in args]
            return self._tcFunctionCall(constructor,
                                        [class_type] + arg_types, exp, env)
            
        # class variable look up
        # usually obj.x, but in eva (prop obj x)
        if self._isOperand('prop', exp):
            self._checkArity(exp, 2)
            obj, x = exp[1:]
            # get obj type from the environment
            obj_type = env.lookup(obj)
            assert isinstance(obj_type, Type.Class),\
              f"self need to be class type, got {obj_type}"
            # see if x is in obj's environment
            return obj_type.getField(x)
        
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

        # arithmatic syntactical sugar
        # (+= x 10) := (set x (+ x 10))
        if self._isOperand(['+=', '-=', '*=', '/='], exp):
            self._checkArity(exp, 2)
            var_name = exp[1]
            var_value = exp[2]
            op = exp[0][0]
            # transform to (set var_name (op var_name var_value)) and type check
            set_exp = ['set', var_name, [op, var_name, var_value]]
            return self.tc(set_exp, env)

        # increment/decrement syntactical sugar
        # (++ x) := (set x (+ x 1))
        if self._isOperand(['++', '--'], exp):
            self._checkArity(exp, 1)
            var_name = exp[1]
            op = exp[0][0]
            value = 1 if op == '+' else -1
            # transform to (set var_name (+ var_name 1)) and type check
            set_exp = ['set', var_name, ['+', var_name, value]]
            return self.tc(set_exp, env)

        # block: sequence of expressions
        # (begin (var x 10) (var y 20) (+ x y))
        if self._isOperand('begin', exp):
            block_env = TypeEnvironment({}, parent=env)
            return self._tcBlock(exp, block_env)

        # if expression: branches must have same type to type check w/o running
        # env |- cond : boolean, env |- then_branch : T, env |- else_branch : T
        # -------------------------------
        # env |- (if cond then_branch else_branch) : T
        if self._isOperand('if', exp):
            self._checkArity(exp, 3)
            cond_type = self.tc(exp[1], env)
            self._expect(cond_type, Type.boolean, exp[1], exp)
            then_type = self.tc(exp[2], env)
            else_type = self.tc(exp[3], env)
            return self._expect(else_type, then_type, exp[3], exp)

        # while loop: condition must be boolean
        # env |- cond : boolean, env |- body : T
        # -------------------------------
        # env |- (while cond body) : T
        if self._isOperand('while', exp):
            self._checkArity(exp, 2)
            cond_type = self.tc(exp[1], env)
            self._expect(cond_type, Type.boolean, exp[1], exp)
            body_type = self.tc(exp[2], env)
            return body_type

        # comparison operators: <, >, <=, >=, ==, !=
        if self._isOperand(['<', '>', '<=', '>=', '==', '!='], exp):
            self._checkArity(exp, 2)
            t1 = self.tc(exp[1], env)
            t2 = self.tc(exp[2], env)
            self._expect(t2, t1, exp[2], exp)
            return Type.boolean

        # lambda function: e.g., (lambda ((x number)) -> number (* x x))
        if self._isOperand('lambda', exp):
            self._checkArity(exp, 4)
            param_list, return_arrow, return_type_str, fn_body = exp[1:]
            if return_arrow != '->':
                raise Exception(
                    f'Syntax error: expected "->" in lambda function "{exp}"')
            return self._tcFunction(param_list, return_type_str, fn_body, env)

        # function definition: e.g., (def sq ((x number)) -> number (* x x))
        # syntactic sugar for (var sq (lambda ((x number)) -> number (* x x)))
        if self._isOperand('def', exp):
            self._checkArity(exp, 5)
            fn_name, param_list, return_arrow, return_type_str, fn_body = exp[
                1:]
            # rewrite as variable declaration with lambda
            new_exp = ['var', fn_name, ['lambda', param_list] + exp[3:]]

            # need to predefine the function name in env for recursion
            param_types = [Type.fromString(param[1]) for param in param_list]
            env.define(
                fn_name,
                FunctionType(param_types, Type.fromString(return_type_str)))

            return self.tc(new_exp, env)

        # function call: e.g. (sq 2)
        if isinstance(exp, list):
            fn_type = self.tc(exp[0], env)
            arg_types = [self.tc(arg, env) for arg in exp[1:]]
            return self._tcFunctionCall(fn_type, arg_types, exp, env)

        raise Exception(f'Unknown expression type: "{exp}"')

    def _tcFunctionCall(self, fn_type: FunctionType,
                        arg_types: list[Type], exp, env) -> Type:
        if not isinstance(fn_type, FunctionType):
            raise Exception(
                f'Type error: expected a function type for "{exp[0]}", but got "{fn_type}" in expression "{exp}"'
            )
        assert len(arg_types) == len(fn_type.param_types), f"arg length mismatch in args_types={arg_types} for function {fn_type} with expected_arg_types={fn_type.param_types}"
        # check arg types
        args = exp[1:]
        for actual_type, expected_type, arg in zip(arg_types,
                                                   fn_type.param_types, args):
            self._expect(actual_type, expected_type, arg, exp)
        return fn_type.return_type

    def _tcFunction(self, param_list, return_type_str, fn_body,
                    env) -> FunctionType:
        # parse param_list
        param_types = []
        param_names = []
        for param in param_list:
            self._checkArity(param, 1)
            param_name = param[0]
            param_type = Type.fromString(param[1])
            param_names.append(param_name)
            param_types.append(param_type)

        return_type = Type.fromString(return_type_str)

        # create a new environment for function body
        fn_env = TypeEnvironment({}, parent=env)
        for param_type, param_name in zip(param_types, param_names):
            fn_env.define(param_name, param_type)

        body_type = self.tc(fn_body, fn_env)
        self._expect(body_type, return_type, fn_body, fn_body)

        return FunctionType(param_types, return_type)

    def _tcBlock(self, exp, env):
        assert exp[0] == 'begin', 'exp should be a begin block'
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

    def _isVariableName(self, exp) -> bool:
        return isinstance(exp, str) and re.match(r'^[a-zA-Z_][a-zA-Z0-9_\.]*$',
                                                 exp)

    def _binary(self, exp, env) -> Type:
        self._checkArity(exp, 2)
        t1 = self.tc(exp[1], env)
        t2 = self.tc(exp[2], env)

        allowed_types = self._getAllowedTypes(exp[0])
        self._expctOperatorType(t1, allowed_types, exp)
        self._expctOperatorType(t2, allowed_types, exp)

        return self._expect(t2, t1, exp[2], exp)

    def _expctOperatorType(self, actualType: Type, allowedTypes: list[Type],
                           exp):
        if actualType not in allowedTypes:
            raise Exception(
                f'Type error: expected one of "{allowedTypes}", got "{actualType}" in expression "{exp}"'
            )

    def _getAllowedTypes(self, op) -> list[Type]:
        if op in ['+', '-', '*', '/']:
            return {
                '+': [Type.number, Type.string],
                '-': [Type.number],
                '*': [Type.number],
                '/': [Type.number],
            }[op]
        raise Exception(f'Unknown operator: "{op}"')

    def _expect(self, actualType, expectedType, value, exp) -> Type:
        if not actualType == expectedType:
            self._throw(actualType, expectedType, value, exp)
        return actualType

    def _throw(self, actualType, expectedType, value, exp):
        raise Exception(
            f'Type error: expected `{expectedType}` for value `{value}` in expression `{exp}`, but got `{actualType}`'
        )

    def _isBinaryOp(self, exp) -> bool:
        return isinstance(exp, list) and exp[0] in ['+', '-', '*', '/']

    def _checkArity(self, exp, expected_arity):
        if len(exp) - 1 != expected_arity:
            raise Exception(
                f'Arity error: expected {expected_arity}, got {len(exp) - 1} in expression {exp}'
            )


def exec(eva, exp):
    if isinstance(exp, str):
        # add (begin ...) to make it a block so that globally I can just write
        # a sequence of expressions
        exp = parser.EvaParser.parse(f'(begin {exp})')

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
    test(eva, ['begin', ['var', 'x', 10], ['var', 'y', 20], ['+', 'x', 'y']],
         Type.number)

    # block: local variables
    test(eva, [
        'begin', ['var', 'x', 10], ['var', 'y', 20],
        ['begin', ['var', 'x', '"hello"'], ['+', 'x', '" world"']],
        ['+', 'x', 'y']
    ], Type.number)

    # block: accessing outer scope variable
    test(eva, ['begin', ['var', 'x', 10], ['begin', ['+', 'x', 5]]],
         Type.number)

    # parsing
    test(eva, '(var x 10) (var y 20)', Type.number)
    test(eva, '(begin (var x 10) (+ x 20))', Type.number)
    # test(eva, '(var x 10) (set x "hello")', Type.string)

    # control flow
    test(eva, 'true', Type.boolean)
    test(eva, 'false', Type.boolean)
    # test(eva, '(< "hello" x)', Type.boolean)
    test(
        eva, '''
         (var x 10)
         (var y 20)
         (if (< x 10)
           2
           1)
         y
         ''', Type.number)

    # while loop
    test(
        eva, '''
         (var x 0)
         (var sum 0)
         (begin
           (set sum 0)
           (while (!= x 10)
             (begin
               #syntactical sugar: (+= sum x) := (set sum (+ sum x))
               (+= sum x)
               # (++ x) := (set x (+ x 1))
               (++ x)
             )
           )
         )
         sum
         ''', Type.number)

    # function
    test(eva, '''
         (def sq ((x number)) -> number (* x x))
         ''', Type.fromString('(Fn (number) number)'))

    # function call
    test(eva, '''
         (sq 5)
         ''', Type.number)

    # test built in function: say math.sin
    test(eva, '''
           (math.sin 3.14)
           ''', Type.number)

    # test function closure
    test(
        eva, '''
           (def makeAdder ((x number)) -> (Fn (number) number)
               (def adder ((y number)) -> number (+ x y))
           )
           (var add5 (makeAdder 5))
         ''', Type.fromString('(Fn (number) number)'))

    # test having a function returning a function of 2 arguments: not currying
    test(
        eva, '''
           (def makeAdder2 ((x number)) -> (Fn (number number) number)
               (def adder2 ((y number) (z number)) -> number (+ x (+ y z))
               )
           )
           (var add10 (makeAdder2 10))
         ''', Type.fromString('(Fn (number number) number)'))

    # recursive function call
    test(
        eva, '''
           (def fact ((n number)) -> number
               (if (== n 0)
                   1
                   (* n (fact (- n 1)))
               )
           )
           (fact 5)
         ''', Type.fromString('number'))

    # lambda function
    test(eva, '''
           (lambda ((x number)) -> number (* x x))
         ''', Type.fromString('(Fn (number) number)'))
    test(
        eva, '''
           (var square (lambda ((x number)) -> number (* x x)))
           (square 6)
         ''', Type.number)

    # immediate invocation of lambda function
    test(
        eva, '''
           ((lambda ((x number)) -> number (* x x)) 7)
         ''', Type.number)

    # type alias: e.g. (type int number)
    test(
        eva, '''
           (type int number)
           (type ID int)
         ''', Type.number)

    # test type alias usage
    test(eva, '''
           ((lambda ((x ID)) -> int (* x x)) 10)
         ''', Type.ID)  # pylint: disable=E1101

    assert Type.ID == Type.number == Type.int, "type neq"  # pylint: disable=E1101

    # test class type
    test(
         eva,
         '''
         (class Person null
           (begin
             (var (name string) "")
             (var (age int) 0)
             (def constructor ((self Person) (name string) (age number)) -> Person
               (begin
                 (set (prop self name) name)
                 # (set (prop self age) age)
                 self
               )
             )

             (def greet ((self Person)) -> string
                ( + "Hello, my name is " (prop self name) )
             )
           )
         )

         (var p1 (new Person "John" 5))
         ((prop p1 greet) p1)
         ''',
         Type.string
         )


    print(eva.global_env.env['Person'].env)
    # print(parseEva.parse('(begin (def sq (x number) (* x x)) (sq 5))'))

    # introduce a list expression
    # print(parseEva.parse('(var x (list 1 2 3))'))
    # # ['var', 'x', ['list', 1, 2, 3]]

    print("All tests passed.")
