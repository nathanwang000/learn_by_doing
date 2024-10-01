import sys, os
from io import StringIO
from typing import List, Dict, Tuple
from enum import Enum
from functools import partial

# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
from lib.utils import pretty_format_dict, binary_or, binary_and, binary_add, binary_flip, binary_neg, WORD_SIZE
from lib.utils import bin2dec, print_with_lines, dec2bin, WORD_SIZE
from lib.assembler import Assembler, Machine

CommandType = Enum(
    'CommandType', 
    ['C_ARITHMETIC', # page 130, fig 7.5: e.g., add, sub, neg ...
     'C_PUSH', # page 131: push <segment> index, e.g., push argument 0 // stack.push(argument[0])
     'C_POP', # page 131: pop <segment> index, e.g., pop argment 0 // argment[0] = stack.pop()
     'C_LABEL', # page 159: label symbol, marks location in code, scope is within the function
     'C_GOTO',  # page 159: goto label, unconditional jump
     'C_IF', # page 159: if-goto label, pc = label if stack.pop() != 0 else pc + 1, label must be within the same function
     'C_FUNCTION', # page 163: function f k, where k is num local variables
     'C_RETURN', # page 163: return, return control to the caller
     'C_CALL', # page 163: call f n, where f is a function and n is number of arguments
    ]
)

ARITHMETIC_COMMANDS = ['add', 'sub', 'neg', 'eq', 'gt', 'lt', 'and', 'or', 'not']

LanguageType = Enum(
    'LanguageType', 
    ['VM', 'ASSEMBLY']
)

def getCommandType(command: str)->CommandType:
    if command in ARITHMETIC_COMMANDS:
        return CommandType.C_ARITHMETIC
    elif command.startswith('push'):
        return CommandType.C_PUSH
    elif command.startswith('pop'):
        return CommandType.C_POP
    elif command.startswith('label'):
        return CommandType.C_LABEL
    elif command.startswith('goto'):
        return CommandType.C_GOTO
    elif command.startswith('if-goto'):
        return CommandType.C_IF
    elif command.startswith('function'):
        return CommandType.C_FUNCTION
    elif command.startswith('return'):
        return CommandType.C_RETURN
    elif command.startswith('call'):
        return CommandType.C_CALL
    
class Interpretor: 
    # AKA python machine that runs VM code
    '''VM interpretor, essentially a simulator in python
    instead of compiling the code, just run on the fly instead
    this is like the class Machine in assembler

    Here we assume VM only contain functions

    Interpreter is essentially a python machine
    where the hardware (including ROM) is specified by self.machine
    and machine language is python (the generated code is the concatenation
    of all the text in code execution parts of self.advance)
    '''
    def __init__(self, 
                 multi_line_comment_open:str='/*',
                 multi_line_comment_close:str='*/',
                 heap_size:int=5, temp_size:int=5, 
                 max_steps:int=100, verbose:bool=True):
        self.multi_line_comment_open = multi_line_comment_open
        self.multi_line_comment_close = multi_line_comment_close
        self.temp_size = temp_size
        self.heap_size = heap_size
        self.max_steps = max_steps
        self.verbose = verbose
        
    def load(self, vm_fnames: List[str]):
        self.machine = {
            'functions': {"Sys.psuedo_init": ["call Sys.init 0"]},
            'current_function': "Sys.psuedo_init",
            'pc': 0, # line number within current_function
            'stack': [], # page 141 M[256:2048]
            'heap': [0] * self.heap_size, # for storing objects, page 141 M[2048:16383]
            # 'sp': 0 # stack pointer: next top location in stack, page 142 R0, this need to be separately implemneted for hack machine but not for python
            'segments': {
                'argument': 0, # dynamically allocated per function, point to a cell on stack, page 142 R2
                'local': 0, # dynamically allocated per function, point to a cell on stack, page 142 R1
                'static': {}, # shared by all functions in the same .vm file, page 141 M[16:256]
                # 'constant': [], # psshared by all functions
                # 'this': [], # pointer to heap: pointer[0]
                # 'that': [], # pointer to heap: pointer[1]
                'pointer': [0, 0], # pointer to this and that, page 142 M[3:5]
                'temp': [0] * self.temp_size, # shared by all functions, page 142 M[5:13]
            }
        }

        # set up symbol lookup: symbol['current_function']['label'] -> 'pc'
        # b/c label in VM is attached to the function scope
        self.symbol_table = {}
        
        # populate functions and symbol table
        for fname in vm_fnames:
            with open(fname) as f:
                self._first_pass(f.readlines())

        if self.verbose:
            print('functions parsed')
            print(pretty_format_dict(self.machine['functions']))
            print('symbol table')
            print(pretty_format_dict(self.symbol_table))

        # make sure Sys.init is encountered
        assert self.machine['current_function'] in self.machine['functions'], f'{self.machine["current_function"]} need to be in vm code'

    def __repr__(self):
        return f'VM interpretor(Machine=\n{pretty_format_dict(self.machine)}\nSymbolTable=\n{pretty_format_dict(self.symbol_table)}\n)'

    def _first_pass(self, codes: List[str], in_comment:bool=False):
        # parse out all the functions into self.machine['functions'], add labels and function to symbol table,
        # and strip out comments
        # in_comment: are you in multi line comment /* */?
        func_name, func_codes = None, []
        lineno_within_func = 0

        for i, l in enumerate(codes):

            if in_comment:
                if self.multi_line_comment_close in l:
                    # comment closed
                    len_c_close = len(self.multi_line_comment_close)
                    l = l[l.index(self.multi_line_comment_close)+len_c_close:]
                    in_comment = False
                else:
                    # comment continue to open, ignore the line
                    continue
                    
            # sanitize line
            l = l.strip()
            orig_l = l
            if '//' in l: # comment
                l = l[:l.index('//')].strip()
            if self.multi_line_comment_open in l:
                in_comment = not orig_l.endswith(self.multi_line_comment_close)
                l = l[:l.index(self.multi_line_comment_open)].strip()
                
            # scrape function
            ct = getCommandType(l)
            if ct is CommandType.C_FUNCTION:
                if func_name is not None:
                    self.machine['functions'][func_name] = func_codes
                    return self._first_pass(codes[i:])
                else:
                    func_name = l.split()[1]
                    if l != "": 
                        lineno_within_func += 1
                        func_codes.append(l)
            elif ct is CommandType.C_LABEL:
                # 'C_LABEL', # page 159: label symbol, marks location in code, scope is within the function
                assert func_name is not None, f'label {l} need to be within a function'
                label = l.split()[1]
                if func_name not in self.symbol_table:
                    self.symbol_table[func_name] = {}
                self.symbol_table[func_name][label] = lineno_within_func
            else:
                if l != "": 
                    lineno_within_func += 1
                    func_codes.append(l)
                
        if func_name is not None:
            self.machine['functions'][func_name] = func_codes
            return


    def advance(self)->bool:
        func_name, pc = self.machine['current_function'], self.machine['pc']
        codes = self.machine['functions'][func_name]

        if self.verbose:
            print('stack:', self.machine['stack'])
            print(f"argument: {self.machine['segments']['argument']}, local: {self.machine['segments']['local']}, len(stack): {len(self.machine['stack'])}")

        
        if pc >= len(codes):
            # finished execution
            return False

        code = codes[pc]
        ct = getCommandType(code)
        if self.verbose:
            print(f'instruction {func_name}[{pc}]: {code}, type: {ct}')
        
        self.machine['pc'] += 1
        # instruction decoding and then execute
        stack = self.machine['stack']
        heap = self.machine['heap']
        segments = self.machine['segments']
        
        if ct is CommandType.C_ARITHMETIC:
            # 'C_ARITHMETIC', # page 130, fig 7.5: e.g., add, sub, neg ...
            x = self.machine['stack'].pop()
            if code in ['add', 'sub', 'eq', 'gt', 'lt', 'and', 'or']:
                # binary operators, TODO: rewrite some of them to bitwise operations 
                y = self.machine['stack'].pop()
                ret = {
                    'add': partial(binary_add, word_size=WORD_SIZE),
                    'sub': lambda a, b: a - b,
                    'eq': lambda a, b: a == b,
                    'gt': lambda a, b: a > b,
                    'lt': lambda a, b: a < b,
                    'and': partial(binary_and, word_size=WORD_SIZE),
                    'or': partial(binary_or, word_size=WORD_SIZE),
                }[code](y, x) # y, x b/c y is first pushed
            else: 
                # unary operators
                ret = {
                    'neg': partial(binary_neg, word_size=WORD_SIZE),
                    'not': partial(binary_flip, word_size=WORD_SIZE),
                }[code](x)
                    
            self.machine['stack'].append(ret)
        elif ct is CommandType.C_PUSH:
            # 'C_PUSH', # page 131: push <segment> index, e.g., push argument 0 // stack.push(argument[0])
            _, segment, index = code.split()
            index = int(index)
            
            if segment == 'argument':
                # dynamically allocated per function, point to a cell on stack
                val = stack[segments[segment] + index]
            elif segment == 'local':
                # dynamically allocated per function, point to a cell on stack
                val = stack[segments[segment] + index]
            elif segment == 'static':
                # shared by all functions in the same .vm file, page 141 M[16:256]
                val = segments[segment][self.machine['current_function']][index]
            elif segment == 'constant':
                # shared by all functions
                val = index
            elif segment == 'this':
                # pointer to heap: pointer[0]
                val = heap[segment['pointer'][0] + index]
            elif segment == 'that':
                # pointer to heap: pointer[1]
                val = heap[segment['pointer'][1] + index]
            elif segment == 'pointer':
                # pointer to this and that, page 142 M[3:5]
                val = self.machine[segment][index]
            elif segment == 'temp':
                # shared by all functions, page 142 M[5:13]
                val = self.machine[segment][index]
            else:
                assert False, f'unknown segment {segment}'
                
            self.machine['stack'].append(val)
            
        elif ct is CommandType.C_POP:
            # 'C_POP', # page 131: pop <segment> index, e.g., pop argment 0 // argment[0] = stack.pop()
            _, segment, index = code.split()
            index = int(index)

            val = self.machine['stack'].pop()
            if segment == 'argument':
                # dynamically allocated per function, point to a cell on stack
                stack[segments[segment] + index] = val
            elif segment == 'local':
                # dynamically allocated per function, point to a cell on stack
                stack[segments[segment] + index] = val
            elif segment == 'static':
                # shared by all functions in .vm file, page 141 M[16:256]
                static_seg = segments[segment]
                curr_f = self.machine['current_function']
                if curr_f not in static_seg:
                    static_seg[curr_f] = [0] * (index+1)
                segments[segment][index] = val
            elif segment == 'constant':
                # shared by all functions
                assert False, 'cannot pop to constant'
            elif segment == 'this':
                # pointer to heap: pointer[0]
                heap[segment['pointer'][0] + index] = val
            elif segment == 'that':
                # pointer to heap: pointer[1]
                heap[segment['pointer'][1] + index] = val
            elif segment == 'pointer':
                # pointer to this and that, page 142 M[3:5]
                self.machine[segment][index] = val
            elif segment == 'temp':
                # shared by all functions, page 142 M[5:13]
                self.machine[segment][index] = val
            else:
                assert False, f'unknown segment {segment}'

        elif ct is CommandType.C_LABEL:
            # 'C_LABEL', # page 159: label symbol, marks location in code, scope is within the function
            assert False, f'{ct} command should only be encountered in _first_pass, as in this second pass it should have been resolved'
        elif ct is CommandType.C_GOTO:
            # 'C_GOTO',  # page 159: goto label, unconditional jump
            label = code.split()[1]
            assert label in self.symbol_table[self.machine['current_function']], f'{label} not in current function scope'
            self.machine['pc'] = self.symbol_table[self.machine['current_function']][label]
        elif ct is CommandType.C_IF: 
            # 'C_IF', # page 159: if-goto label, pc = label if stack.pop() != 0 else pc + 1, label must be within the same function
            label = code.split()[1]
            label_pc = self.symbol_table[self.machine['current_function']][label]
            if self.machine['stack'].pop() != 0:
                self.machine['pc'] = label_pc # no need to consider = 0 case b/c that's already default
        elif ct is CommandType.C_FUNCTION:
            # 'C_FUNCTION', # page 163: function f k, where k is num local variables
            n_local = int(code.split()[2])
            for _ in range(n_local):
                stack.append(0)
        elif ct is CommandType.C_RETURN:
            # 'C_RETURN', # page 163: return, return control to the caller
            frame = segments['local']
            prev_f, prev_f_lineno = stack[frame-5] # ret_addr for python machine it should be (), but for hack it will be a label to ROM
            # put return value back at the new top: see page 162, here I think the assumption is a function always returns something, could be None
            stack[segments['argument']] = stack.pop()
            # print(frame, len(stack))
            sp = segments['argument'] + 1
            segments['pointer'] = [stack[frame-2], stack[frame-1]]
            segments['argument'] = stack[frame-3]
            segments['local'] = stack[frame-4]
            self.machine['stack'] = self.machine['stack'][:sp]

            # goto prev function
            self.machine['current_function'] = prev_f
            self.machine['pc'] = prev_f_lineno
        elif ct is CommandType.C_CALL:
            # 'C_CALL', # page 163: call f n, where f is a function and n is number of arguments
            _, next_f, n_args = code.split()
            n_args = int(n_args)
            # push return-address
            curr_f, pc = self.machine['current_function'], self.machine['pc']
            stack.append((curr_f, pc))
            # push lcl
            stack.append(segments['local'])
            # pushl arg
            stack.append(segments['argument'])
            # push this and that
            stack.extend(segments['pointer'])
            # arg = SP - n - 5
            segments['argument'] = len(stack) - n_args - 5
            # lcl = SP
            segments['local']= len(stack)
            # goto f
            self.machine['current_function'] = next_f
            self.machine['pc'] = 0
        else:
            assert False, f'command {ct} not found'
        
        return True

    
    def __call__(self, vm_fnames: List[str]):
        self.load(vm_fnames)

        steps = 0
        while steps <= self.max_steps:
            if self.verbose:
                print('Machine step', steps)
            steps += 1
            ok = self.advance()
            if not ok:
                print('Finished execution')
                return
        print(f'Program terminated b/c exceeding max step of {self.max_steps}')        

def lineStrip(code:str)->List[str]:
    return list(filter(lambda x: x != "",
                       map(lambda x: x.strip(), code.split('\n'))))
    
class Translator:
    '''VM code -> Assembly code'''
    def __init__(self, 
                 multi_line_comment_open:str='/*',
                 multi_line_comment_close:str='*/',
                 temp_addr:int=5,
                 verbose:bool=True):
        self.multi_line_comment_open = multi_line_comment_open
        self.multi_line_comment_close = multi_line_comment_close
        self.n_symbol = 0 # used to track number of system symbol assigned
        self.current_function = None # useful for label: func_name$label
        self.temp_addr = temp_addr
        self.verbose = verbose

    def getVMcode(self, fname:str, lineno:int):
        '''return the vm code corresponds to fname and lineno; this handles pseudo code as well'''
        return self.compiledCodes[self.compiledLinenos.index((fname, lineno))][0]
        
    def load(self, vm_fnames: List[str]):
        self.pc = 0 # where we are in the code
        self.sanitizedCodes = {} # fname -> {"codes", "linenos"}

        for fname in vm_fnames:
            with open(fname) as f:
                codes, linenos = self._sanitizeCodes(f.readlines())
                self.sanitizedCodes[fname] = {"codes": codes, "linenos": linenos}

        # insert initialization code, call Sys.init with no args
        # this is intermingled with direct assembly code for setting up 
        # utilities like symbols for setting up == operator
        self.compiledCodes = [
            # setup SP value to 256
            ('@256', LanguageType.ASSEMBLY),
            ('D=A', LanguageType.ASSEMBLY),
            ('@SP', LanguageType.ASSEMBLY),
            ('M=D', LanguageType.ASSEMBLY),
            # call initialization
            ("call Sys.init 0", LanguageType.VM),
            # jump to the end
            ("@0END", LanguageType.ASSEMBLY),
            ("0;JMP", LanguageType.ASSEMBLY),
        ]
        self.compiledLinenos = [("psuedo code file", i) for i in range(len(self.compiledCodes))]
        psuedo_lines = len(self.compiledCodes)

        # add back middle code
        for k, v in self.sanitizedCodes.items():
            self.compiledCodes.extend(
                zip(v["codes"], [LanguageType.VM] * len(v["codes"]))
            )
            for ln in v["linenos"]:
                self.compiledLinenos.append((k, ln))

        # jump to end of the code
        end_codes = [
            # start with number to differentiate from user label
            ("(0END)", LanguageType.ASSEMBLY),
        ]
        self.compiledCodes.extend(end_codes)
        self.compiledLinenos.extend([("psuedo code file", psuedo_lines+i) for i in range(len(end_codes))])

        if self.verbose:
            print('code sanitized')
            print('==============')
            print(pretty_format_dict(self.sanitizedCodes))
            print('compiled code')
            print('==============')
            print(self)
            print('==============')

    def __repr__(self):
        ret = ['VM translator(']
        for (code, langType), (fn, ln) in zip(self.compiledCodes, self.compiledLinenos):
            ret.append(f'({fn}, {ln}, {langType}): {code}')
        return "\n".join(ret + [')'])

    def _sanitizeCodes(self, codes: List[str])->Tuple[List[str], List[int]]:
        '''
        strip out comments e.g. // or /* comment */
        return (a list of command that are sanitized without comments, linenos in orignal code)
        '''
        def helper(codes:List[str], lineno:int, in_comment:bool, result: Tuple[List[str], List[int]])->Tuple[List[str], List[int]]:
            if len(codes) == 0:
                return result
            l, rest = codes[0], codes[1:]

            if in_comment:
                if self.multi_line_comment_close in l:
                    # comment closed
                    len_c_close = len(self.multi_line_comment_close)
                    l = l[l.index(self.multi_line_comment_close)+len_c_close:]
                    in_comment = False
                else:
                    # comment continue to open, ignore the line
                    return helper(rest, lineno+1, True, result)

            # sanitize line
            l = l.strip()
            orig_l = l
            if '//' in l: # comment
                l = l[:l.index('//')].strip()
            if self.multi_line_comment_open in l:
                in_comment = not orig_l.endswith(self.multi_line_comment_close)
                l = l[:l.index(self.multi_line_comment_open)].strip()

            if l == "":
                return helper(rest, lineno+1, in_comment, result)

            sanitized_codes, linenos = result
            return helper(rest, lineno+1, in_comment, (sanitized_codes + [l], linenos + [lineno]))
                        
        return helper(codes, 0, False, ([], []))

    def assignSystemSymbol(self)->int:
        self.n_symbol += 1
        return self.n_symbol - 1

    def _assPushD(self)->List[str]:
        '''push D to stack'''
        return lineStrip(
            '''
            @SP
            A=M
            M=D // M[M[SP]] is the tip of stack
            // M[SP] += 1
            @SP
            M=M+1
            '''
        )
        
    def _assPushA(self, s:str)->List[str]:
        '''push A=@s to stack in assembly code'''
        
        return [
            f'@{s}',
            'D=A',
        ] + self._assPushD()

    def _assPopToD(self)->List[str]:
        'pop stack to D'
        return lineStrip(
            '''
            // D=stack.pop()
            @SP
            A=M-1
            D=M // D=M[M[SP]-1] that is stack.top()
            @SP
            M=M-1 // M[SP] -= 1 // D=stack.pop()
            '''
        )
        
    def _assArithmetic(self, code:str)->List[str]:
        "given vm command output assembly codes"
        # 'C_ARITHMETIC', # page 130, fig 7.5: e.g., add, sub, neg ...
        # temp[1] = self.machine['stack'].pop()
        ass_codes = []
        temp = self.temp_addr
        ass_codes.extend(
            self._assPopToD() + lineStrip(
                f'''
                // M[temp+1]=D
                @{temp+1}
                M=D
                '''
            )
        )

        assert code in ARITHMETIC_COMMANDS, 'code invalid'
        if code in ['add', 'sub', 'eq', 'gt', 'lt', 'and', 'or']:
            # binary operators
            # temp[0] = self.machine['stack'].pop(), use temp[0] to store the first pushed element b/c stack is FILO
            ass_codes.extend(
                self._assPopToD() + lineStrip(
                    f'''
                    // M[temp+0]=D
                    @{temp+0}
                    M=D
                    '''
                )
            )

            # comput and push to stack
            if code == 'add':
                ass_codes.extend(
                    lineStrip(
                        f'''
                        //* stack.push(temp[0] + temp[1])
                        @{temp+0}
                        D=M
                        @{temp+1}
                        D=D+M
                        '''
                    ) + self._assPushD())
                
            elif code == 'sub':
                ass_codes.extend(
                    lineStrip(
                        f'''
                        //* stack.push(temp[0] - temp[1])
                        @{temp+0}
                        D=M
                        @{temp+1}
                        D=D-M
                        '''
                    ) + self._assPushD())
            elif code == 'eq': 
                ass_codes.extend(
                    lineStrip(
                        f'''
                        // temp[0]-temp[1] == 0
                        @{temp+0}
                        D=M
                        @{temp+1}
                        D=D-M
                        // test if eq 0
                        @{self.assignSystemSymbol()}ADD_TRUE
                        D;JEQ
                        // case for not true: push 0 and continue
                        '''
                    ) + self._assPushA('0') + lineStrip(
                        f'''
                        @{self.assignSystemSymbol()}cont
                        0;JMP
                        // case for true: push 0xF to top
                        ({self.n_symbol-2}ADD_TRUE)  
                        '''
                    ) + self._assPushA(bin2dec('1'*WORD_SIZE)) + lineStrip(
                        f'''
                        // continue
                        ({self.n_symbol-1}cont)
                        '''
                    ))
            elif code == 'gt': 
                ass_codes.extend(
                    lineStrip(
                        f'''
                        // temp[0]-temp[1] > 0
                        @{temp+0}
                        D=M
                        @{temp+1}
                        D=D-M
                        // test if > 0
                        @{self.assignSystemSymbol()}ADD_TRUE
                        D;JGT
                        // case for not true: push 0 and continue
                        '''
                    ) + self._assPushA('0') + lineStrip(
                        f'''
                        @{self.assignSystemSymbol()}cont
                        0;JMP
                        // case for true: push 0xF to top
                        ({self.n_symbol-2}ADD_TRUE)  
                        '''
                    ) + self._assPushA(bin2dec('1'*WORD_SIZE)) + lineStrip(
                        f'''
                        // continue
                        ({self.n_symbol-1}cont)
                        '''
                    ))
            elif code == 'lt': 
                ass_codes.extend(
                    lineStrip(
                        f'''
                        // temp[0]-temp[1] < 0
                        @{temp+0}
                        D=M
                        @{temp+1}
                        D=D-M
                        // test if < 0
                        @{self.assignSystemSymbol()}ADD_TRUE
                        D;JLT
                        // case for not true: push 0 and continue
                        '''
                    ) + self._assPushA('0') + lineStrip(
                        f'''
                        @{self.assignSystemSymbol()}cont
                        0;JMP
                        // case for true: push 0xF to top
                        ({self.n_symbol-2}ADD_TRUE)  
                        '''
                    ) + self._assPushA(bin2dec('1'*WORD_SIZE)) + lineStrip(
                        f'''
                        // continue
                        ({self.n_symbol-1}cont)
                        '''
                    ))
            elif code == 'and': 
                ass_codes.extend(
                    lineStrip(
                        f'''
                        // M[temp + 0] & M[temp + 1] and push to stack
                        @{temp+0}
                        D=M
                        @{temp+1}
                        D=D&M
                        '''
                    ) + self._assPushD()
                )
            elif code == 'or': 
                ass_codes.extend(
                    lineStrip(
                        f'''
                        // M[temp + 0] | M[temp + 1] and push to stack
                        @{temp+0}
                        D=M
                        @{temp+1}
                        D=D|M
                        '''
                    ) + self._assPushD()
                )
        else: 
            # unary operators
            if code == 'neg':
                ass_codes.extend(
                    lineStrip(
                        f'''
                        // -M[temp + 1]
                        @{temp+1}
                        D=-M
                        '''
                    ) + self._assPushD()
                )
            elif code == 'not':
                ass_codes.extend(
                    lineStrip(
                        f'''
                        // !M[temp + 1]
                        @{temp+1}
                        D=!M
                        '''
                    ) + self._assPushD()
                )
        return ass_codes

    def _assFunc(self, code:str)->List[str]:
        "given vm command output assembly codes"
        # 'C_FUNCTION', # page 163: function f k, where k is num local variables
        n_local = int(code.split()[2])
        func_name = code.split()[1]
        ass_codes = [f'({func_name})']
        for _ in range(n_local):
            ass_codes.extend(self._assPushA('0'))
        return ass_codes

    def _assCall(self, command:str)->List[str]:
        "given vm command output assembly codes"
        # # 'C_CALL', # page 163: call f n, where f is a function and n is number of arguments
        ass_codes = []
        _, next_f, n_args = command.split()
        n_args = int(n_args)
        # push return-address
        ass_codes.extend(self._assPushA(f'{self.assignSystemSymbol()}return-address'))
        # push M[lcl]
        ass_codes.extend(
            lineStrip(
                '''
                @LCL
                D=M
                '''
            ) + self._assPushD())
        # pushl M[arg]
        ass_codes.extend(
            lineStrip(
                '''
                @ARG
                D=M
                '''
            ) + self._assPushD())
        # push M[this] and M[that]
        ass_codes.extend(
            lineStrip(
                '''
                @THIS
                D=M
                '''
            ) + self._assPushD())
        ass_codes.extend(
            lineStrip(
                '''
                @THAT
                D=M
                '''
            ) + self._assPushD())
        # arg = SP - n_args - 5; 5 b/c we just pushed 5 elements
        ass_codes.extend([
            '@SP',
            'D=M',
            f'@{n_args}',
            'D=D-A',
            '@5',
            'D=D-A',
            '@ARG',
            'M=D',
        ])
        # lcl = SP; set local to the new function's local segment
        ass_codes.extend([
            '@SP',
            'D=M',
            '@LCL',
            'M=D',
        ])
        # goto f
        ass_codes.extend([
            f'@{next_f}',
            '0;JMP',
        ])
        ass_codes.append(f'({self.n_symbol-1}return-address)')
        return ass_codes

    def _assPop(self, code:str, fname:str)->List[str]:
        # 'C_POP', # page 131: pop <segment> index, e.g., pop argment 0 // argment[0] = stack.pop()
        _, segment, index = code.split()
        index = int(index)
        ret = self._assPopToD()
        
        if segment == 'argument':
            # dynamically allocated per function, point to a cell on stack
            ret += lineStrip(
                f'''
                // M[M[ARG]+index] = stack.pop()
                @{self.temp_addr+0}
                M=D
                @{index}
                D=A
                @ARG
                D=D+M // M[ARG]+index
                @{self.temp_addr+1}
                M=D // temp[1] = M[ARG]+index
                @{self.temp_addr+0}
                D=M // retrieve saved top
                @{self.temp_addr+1}
                A=M
                M=D
                '''
            )
        elif segment == 'local':
            # dynamically allocated per function, point to a cell on stack
            ret += lineStrip(
                f'''
                // M[M[LCL]+index] = stack.pop()
                @{self.temp_addr+0}
                M=D
                @{index}
                D=A
                @LCL
                D=D+M // M[LCL]+index
                @{self.temp_addr+1}
                M=D // temp[1] = M[LCL]+index
                @{self.temp_addr+0}
                D=M // retrieve saved top
                @{self.temp_addr+1}
                A=M
                M=D
                '''
            )
        elif segment == 'static':
            # shared by all functions in .vm file, page 141 M[16:256]            
            ret += lineStrip(
                f'''
                // M[fname.index] = stack.pop()
                @{fname}.{index}
                M=D
                '''
            )
        elif segment == 'constant':
            # shared by all functions
            assert False, 'cannot pop to constant'
        elif segment == 'this':
            ret += lineStrip(
                f'''
                // M[M[THIS]+index] = stack.pop()
                @{self.temp_addr+0}
                M=D
                @{index}
                D=A
                @THIS
                D=D+M // M[THIS]+index
                @{self.temp_addr+1}
                M=D // temp[1] = M[THIS]+index
                @{self.temp_addr+0}
                D=M // retrieve saved top
                @{self.temp_addr+1}
                A=M
                M=D
                '''
            )
        elif segment == 'that':
            ret += lineStrip(
                f'''
                // M[M[THAT]+index] = stack.pop()
                @{self.temp_addr+0}
                M=D
                @{index}
                D=A
                @THAT
                D=D+M // M[THAT]+index
                @{self.temp_addr+1}
                M=D // temp[1] = M[LCL]+index
                @{self.temp_addr+0}
                D=M // retrieve saved top
                @{self.temp_addr+1}
                A=M
                M=D
                '''
            )
        elif segment == 'pointer':
            # pointer[0] and pointer[1] are this and that, page 142 M[3:5]
            ret += lineStrip(
                f'''
                // M[THIS+index] = stack.pop()
                @{self.temp_addr+0}
                M=D
                @{index}
                D=A
                @THIS
                D=D+A // THIS +index
                @{self.temp_addr+1}
                M=D // temp[1] = THIS +index
                @{self.temp_addr+0}
                D=M // retrieve saved top
                @{self.temp_addr+1}
                A=M
                M=D
                '''
            )
        elif segment == 'temp':
            # shared by all functions, page 142 M[5:13]
            ret += lineStrip(
                f'''
                // M[temp+index] = stack.pop()
                @{self.temp_addr+index}
                M=D
                '''
            )
        else:
            assert False, f'unknown segment {segment}'

        return ret

    def _assPush(self, code:str, fname:str)->List[str]:
        "given vm code output assembly codes; fname is the file where the command is from"
        # 'C_PUSH', # page 131: push <segment> index, e.g., push argument 0 // stack.push(argument[0])
        _, segment, index = code.split()
        index = int(index)

        # load value into D
        if segment == 'argument':
            # dynamically allocated per function, point to a cell on stack
            # D = M[M[ARG] + index]
            ret = lineStrip(
                    f'''
                    @{index}
                    D=A
                    @ARG
                    A=D+M
                    // A is M[ARG] + index
                    D=M
                    '''
            )
        elif segment == 'local':
            # dynamically allocated per function, point to a cell on stack
            # D = M[M[LCL] + index]
            ret = lineStrip(
                    f'''
                    @{index}
                    D=A
                    @LCL
                    A=D+M
                    // A is M[LCL] + index
                    D=M
                    '''
            )
        elif segment == 'static':
            # shared by all functions in XXX.vm file, page 141 M[16:256]
            # D = M[XXX.{index}]
            ret = lineStrip(
                    f'''
                    @{fname}.{index}
                    D=M
                    '''
            )
        elif segment == 'constant':
            # shared by all functions
            # D = index
            ret = [f'@{index}', 'D=A']
        elif segment == 'this':
            # pointer to heap: THIS
            # D = M[M[THIS] + index]
            ret = lineStrip(
                    f'''
                    @{index}
                    D=A
                    @THIS
                    A=D+M
                    // A is M[THIS] + index
                    D=M
                    '''
            )
        elif segment == 'that':
            # pointer to heap: THAT
            # D = M[M[THAT] + index]
            ret = lineStrip(
                    f'''
                    @{index}
                    D=A
                    @THAT
                    A=D+M
                    // A is M[THAT] + index
                    D=M
                    '''
            )
        elif segment == 'pointer':
            # pointer to this and that, page 142 M[3:5]
            # D = M[THIS+index]
            ret = lineStrip(
                    f'''
                    @{index}
                    D=A
                    @THIS
                    A=D+A
                    D=M
                    '''
            )
        elif segment == 'temp':
            # shared by all functions, page 142 M[5:13]
            # D = M[R5+index]
            ret = lineStrip(
                    f'''
                    @{index}
                    D=A
                    @{self.temp_addr}
                    A=D+A
                    D=M
                    '''
            )
        else:
            assert False, f'unknown segment {segment}'
            
        return ret + self._assPushD()

    def _assReturn(self)->List[str]:
        # 'C_RETURN', # page 163: return, return control to the caller
        # stack looks like [arguments, prev_ret_addr, prev_local, prev_arg, prev_this, prev_that, lcl]
        # the new SP will be ARG+1 b/c ARG is where new returned value is
        return lineStrip(
            f'''
            // M[temp+0] = M[LCL]
            @LCL
            D=M
            @{self.temp_addr+0}
            MD=D
            // M[temp+1] = M[M[temp+0]-5] ; save return addr
            @5
            A=D-A // M[temp+0]-5
            D=M // M[M[temp+0]-5]
            @{self.temp_addr+1}
            M=D
            '''
        ) + self._assPopToD() + lineStrip(
            f'''
            // M[M[ARG]] = stack.pop(); pop the return value to arg segment
            @ARG
            A=M 
            M=D // M[M[ARG]] = poped value
            // M[SP] = M[ARG] + 1
            @ARG
            D=M+1
            @SP
            M=D
            // M[THAT] = M[M[TEMP+0]-1]
            @{self.temp_addr+0}
            A=M-1
            D=M
            @THAT
            M=D
            // M[THIS] = M[M[TEMP+0]-2]
            @2
            D=A
            @{self.temp_addr+0}
            A=M-D
            D=M
            @THIS
            M=D
            // M[ARG] = M[M[TEMP+0]-3]
            @3
            D=A
            @{self.temp_addr+0}
            A=M-D
            D=M
            @ARG
            M=D
            // M[LCL] = M[M[TEMP+0]-4]
            @4
            D=A
            @{self.temp_addr+0}
            A=M-D
            D=M
            @LCL
            M=D
            // goto M[temp+1]
            @{self.temp_addr+1}
            A=M
            0;JMP
            '''
        )

    def _assIf(self, code:str, func_name:str)->List[str]:
        # 'C_IF', # page 159: if-goto label, pc = label if stack.pop() != 0 else pc + 1, label must be within the same function
        label = code.split()[1]
        return self._assPopToD() + lineStrip(
            f'''
            @{func_name}${label}
            D;JNE
            '''
        )
        
    def _assGoto(self, code:str, func_name:str)->List[str]:
        # 'C_GOTO',  # page 159: goto label, unconditional jump
        label = code.split()[1]
        return lineStrip(
            f'''
            @{func_name}${label}
            0;JMP
            '''
        )

    def _assLabel(self, code:str, func_name:str)->List[str]:
        # 'C_LABEL', # page 159: label symbol, marks location in code, scope is within the function
        label = code.split()[1]
        return lineStrip(
            f'''
            ({func_name}${label})
            '''
        )
        
    def advance(self)->Tuple[bool, List[str], Tuple[str, int]]:
        '''
        return (ok, assembly codes for the next command, reference_line_in_orig_file)

        references in book:
        'pc': 0, # line number within current_function
        'stack': [], # page 141 M[256:2048]
        'heap': [0]*self.heap_size, # for storing objects, page 141 M[2048:16383]
        'sp': 0 # stack pointer: next top location in stack, page 142 R0
        'segments': {
            'argument': 0, # dynamically allocated per function, point to a cell on stack, page 142 R2
            'local': 0, # dynamically allocated per function, point to a cell on stack, page 142 R1
            'static': [0] * self.static_size, # shared by all functions in .vm file, page 141 M[16:256]
            'constant': [], # psshared by all functions
            'this': [], # pointer to heap: pointer[0]
            'that': [], # pointer to heap: pointer[1]
            'pointer': [0, 0], # pointer to this and that, page 142 M[3:5]
            'temp': [0] * self.temp_size, # shared by all functions, page 142 M[5:13]
        '''

        if self.pc >= len(self.compiledCodes):
            # finished execution
            return False, [""], ("", 0)

        # instruction fetching
        code, langType = self.compiledCodes[self.pc]
        fname, code_lineno = self.compiledLinenos[self.pc]
        ct = getCommandType(code)
        self.pc += 1        
        if self.verbose:
            print(f'instruction from {fname}[{code_lineno}]: {code}, type: {ct}')

        if langType == LanguageType.ASSEMBLY:
            return True, [code], (fname, code_lineno)

        # instruction decoding and then execute
        ass_codes = []
        if ct is CommandType.C_ARITHMETIC:
            # page 130: add, neg, ...
            ass_codes.extend(self._assArithmetic(code))
        elif ct is CommandType.C_PUSH:
            # page 131: push <segment:e.g., local> <index: e.g., 1>
            ass_codes.extend(self._assPush(code, fname))
        elif ct is CommandType.C_POP:
            # page 131: pop <segment> index
            ass_codes.extend(self._assPop(code, fname))
        elif ct is CommandType.C_LABEL:
            # page 159: label symbol
            ass_codes.extend(self._assLabel(code, self.current_function))
        elif ct is CommandType.C_GOTO:
            # page 159: goto label
            ass_codes.extend(self._assGoto(code, self.current_function))
        elif ct is CommandType.C_IF:
            # if-goto label
            ass_codes.extend(self._assIf(code, self.current_function))
        elif ct is CommandType.C_FUNCTION:
            # page 163: function f k
            self.current_function = code.split()[1]
            ass_codes.extend(self._assFunc(code))
        elif ct is CommandType.C_RETURN:
            # page 163: return
            ass_codes.extend(self._assReturn())
        elif ct is CommandType.C_CALL:
            # page 163: call f n
            ass_codes.extend(self._assCall(code))
        else:
            assert False, f'command {ct} not founc'

        return True, ass_codes, (fname, code_lineno)
    
    def __call__(self, vm_fnames: List[str])->Tuple[str, List[Tuple[str, int]]]:
        '''
        given vm code in vm_fnames, 
        return (assembly_code, vm_code_lines that corresponds to assembly_code for debug)
        '''
        self.load(vm_fnames)

        codes = []
        linenos = []
        while True:
            ok, ass_codes, lineno = self.advance()
            if not ok: break
            codes.append(ass_codes)
            linenos.append(lineno)

        # build assembly line to vm line dictionary
        tgtRef2srcRef = []
        for (fname, i), codeLines in zip(linenos, codes):
            tgtRef2srcRef.extend([(fname, i)] * len(codeLines))
            
        return '\n'.join(['\n'.join(codeLines) for codeLines in codes]), tgtRef2srcRef