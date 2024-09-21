import sys
import os
from io import StringIO
from typing import Tuple, List, Dict
from enum import Enum

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.utils import pretty_format_dict, dec2bin, bin2dec, isInt, print_with_lines
from lib.utils import binary_flip, binary_and, binary_or, binary_add, binary_neg, WORD_SIZE

CommandType = Enum('CommandType', 
                   ['A_COMMAND', # @symbol
                    'C_COMMAND', # dest=comp;jump
                    'L_COMMAND'] # (symbol)
                  )

class Code:
    _dest_lists = ['null', 'M', 'D', 'MD', 'A', 'AM', 'AD', 'AMD']
    _jump_lists = ['null', 'JGT', 'JEQ', 'JGE', 'JLT', 'JNE', 'JLE', 'JMP']
    dest = {name:dec2bin(i, 3) for i, name in enumerate(_dest_lists)}
    jump = {name:dec2bin(i, 3) for i, name in enumerate(_jump_lists)}
    comp = {
        '0': '0101010',
        '1': '0111111',
        '-1': '0111010',
        'D': '0001100',
        'A': '0110000',
        '!D': '0001101',
        '!A': '0110001',
        '-D': '0001111',
        '-A': '0110011',
        'D+1': '0011111',
        'A+1': '0110111',
        'D-1': '0001110',
        'A-1': '0110010',
        'D+A': '0000010',
        'D-A': '0010011',
        'A-D': '0000111',
        'D&A': '0000000',
        'D|A': '0010101',
        'M': '1110000',
        '!M': '1110001',
        '-M': '1110011',
        'M+1': '1110111',
        'M-1': '1110010',
        'D+M': '1000010',
        'D-M': '1010011',
        'M-D': '1000111',
        'D&M': '1000000',
        'D|M': '1010101',
    }
    compCode2mnemonic = {v: k for k, v in comp.items()}
    jumpCode2mnemonic = {v: k for k, v in jump.items()}
    destCode2mnemonic = {v: k for k, v in dest.items()}
    def compFun(comp_code):

        assert comp_code in Code.comp.values()

        def _f(A:int, M:int, D:int)->int:
            'A/D: A/D register value, M: RAM[A] value'
            # TODO: this version doesn't simulate overflow as it can compute value more than 1 word
            # TODO: replace this with a ALU to really simulate the hardware
            mnemonic2output = {
                '0': 0,
                '1': 1,
                '-1': binary_neg(1, WORD_SIZE),
                'D': D,
                'A': A,
                '!D': binary_flip(D, WORD_SIZE),
                '!A': binary_flip(A, WORD_SIZE),
                '-D': binary_neg(D, WORD_SIZE),
                '-A': binary_neg(A, WORD_SIZE),
                'D+1': binary_add(D, 1, WORD_SIZE),
                'A+1': binary_add(A, 1, WORD_SIZE),
                'D-1': binary_add(D, binary_neg(1, WORD_SIZE), WORD_SIZE),
                'A-1': binary_add(A, binary_neg(1, WORD_SIZE), WORD_SIZE),
                'D+A': binary_add(D, A, WORD_SIZE),
                'D-A': binary_add(D, binary_neg(A, WORD_SIZE), WORD_SIZE),
                'A-D': binary_add(A, binary_neg(D, WORD_SIZE), WORD_SIZE),
                'D&A': binary_and(D, A, WORD_SIZE),
                'D|A': binary_or(D, A, WORD_SIZE),
                'M': M,
                '!M': binary_flip(M, WORD_SIZE),
                '-M': binary_neg(M, WORD_SIZE),
                'M+1': binary_add(M, 1, WORD_SIZE),
                'M-1': binary_add(M, binary_neg(1, WORD_SIZE), WORD_SIZE),
                'D+M': binary_add(D, M, WORD_SIZE),
                'D-M': binary_add(D, binary_neg(M, WORD_SIZE), WORD_SIZE),
                'M-D': binary_add(M, binary_neg(D, WORD_SIZE), WORD_SIZE),
                'D&M': binary_and(D, M, WORD_SIZE),
                'D|M': binary_or(D, M, WORD_SIZE),
            }
            return mnemonic2output[Code.compCode2mnemonic[comp_code]]

        return _f
        
class SymbolTable:
    def __init__(self):
        self.d = {
            # page 110 of 'the elements of computing systems'
            # symbol to RAM address
            'SP': 0,
            'LCL': 1,
            'ARG': 2,
            'THIS': 3,
            'THAT': 4, 
            'SCREEN': 16384,
            'KBD': 24576
        }
        for i in range(16): # R0-R15
            self.d[f'R{i}'] = i

    def addEntry(self, symbol:str, address:int):
        assert symbol not in self.d, f'{symbol} already in symbol table'
        self.d[symbol] = address

    def contains(self, symbol:str):
        return symbol in self.d

    def getAddress(self, symbol:str):
        return self.d[symbol]

    def __repr__(self):
        ret = f'{"symbol":10s}|{"address":10s}\n' + '-' * 21
        for k, v in self.d.items():
            ret += f'\n{k:10s}|{str(v):10s}'
        return ret

class Parser:

    def __init__(self, fs: StringIO):
        # fs is stream of assembly code
        """
        parse assembly code
        >>> assembly_example = '''
        // comment
        @a
        D=D+M
        @100
        D=A // load A into D
        '''
        >>> p = Parser(StringIO(assembly_example))
        >>> while True:
        >>>    ok, command = p.advance()
        >>>    if not ok: break
        >>>    ct = p.commandType(command)
        >>>    if ct == CommandType.C_COMMAND:
        >>>        print(f'{command:10s} {ct:10s} (dest: {p.dest(command)} = comp: {p.comp(command)}; jump: {p.jump(command)})')
        >>>    else:
        >>>        print(f'{command:10s} {ct:10s} (symbol: {p.symbol(command)})')
        """
        self.filestream = fs
        self.machine_code_lineno = 0 # machine code line number
        self.ass_code_lineno = 0 # assmebly code line number
        
    def advance(self)->Tuple[bool, str, int]:
        'return (ok, next command, reference_line_in_orig_file)'
        l = self.filestream.readline()
        if l == '':
            # EOF error
            return False, '', self.ass_code_lineno

        self.ass_code_lineno += 1
        l = l.strip()
        if '//' in l:
            l = l[:l.index('//')].strip()
            
        if l == '':    
            return self.advance()
        else:
            if not l.startswith('('):
                # pseudo command doesn't advance lineno
                self.machine_code_lineno += 1
            # ass_code_lineno - 1 to restore the original value
            # before advance
            return True, l, self.ass_code_lineno - 1

    def commandType(self, command):
        if command.startswith('@'):
            return CommandType.A_COMMAND
        elif command.startswith('('):
            return CommandType.L_COMMAND
        else:
            return CommandType.C_COMMAND

    def symbol(self, command):
        assert self.commandType(command) != CommandType.C_COMMAND, 'symbol does not apply to C command'
        if command.startswith('@'):
            return command[1:].strip()
        else:
            assert command[-1] == ')', 'L_COMMAND must end i )'
            return command[1:-1].strip()

    def dest(self, command):
        assert self.commandType(command) == CommandType.C_COMMAND, 'dest only apply to C command'
        o = 'null'
        if '=' in command:
            o = command.split('=')[0]
            assert o in Code.dest, f'{o} not in dest {Code.dest.keys()}'
        return o

    def comp(self, command):
        assert self.commandType(command) == CommandType.C_COMMAND, 'comp only apply to C command'
        o = command
        if ';' in command:
            o = o.split(';')[0]
        if '=' in command:
            o = o.split('=')[1]
        assert o in Code.comp, f'{o} not in comp {Code.comp.keys()}'
        return o

    def jump(self, command):
        assert self.commandType(command) == CommandType.C_COMMAND, 'jump only apply to C command'
        o = 'null'
        if ';' in command:
            o = command.split(';')[1]
        assert o in Code.jump, f'{o} not in jump {Code.jump.keys()}'
        return o    

class Machine: # hack machine
    def __init__(self, memory_size:int=100, max_steps:int=100, verbose:bool=False):
        self.memory_size = memory_size
        self.max_steps = max_steps # max runtime allowed
        self.verbose = verbose
        
    def load(self, machine_code, ass_linenos=None):
        codes = machine_code.split('\n')
        self.memory_size = max(self.memory_size, len(codes))
        self.machine = {
            'PC': 0, # program counter
            'A': 0, # A register
            'D': 0, # D register
            'ROM': codes, # code memory
            'RAM': [0] * self.memory_size,
        }

        if ass_linenos is not None:
            # useful for debugging to show machine code's corresponding assembly line numbers
            self.machine['assembly_linenos'] = ass_linenos
            
    def _isA(self, command:str):
        # is A command, if False then C command
        # see specification on page 109
        return command[0] == '0'

    def _comp(self, command:str):
        # see specification on page 109
        assert not self._isA(command)
        return command[3:-6]

    def _dest(self, command:str):
        # see specification on page 109
        assert not self._isA(command)
        return command[-6:-3]

    def _jump(self, command:str):
        # see specification on page 109
        assert not self._isA(command)
        return command[-3:]    
        
    def advance(self)->bool:
        ## fetch instruction
        if self.machine['PC'] >= len(self.machine['ROM']):
            # finishes execution
            return False
        command = self.machine['ROM'][self.machine['PC']]

        if self.verbose:
            log = f"instr {self.machine['PC']}: "
            if self._isA(command):
                log += f'@{bin2dec(command[1:])}'
            else:
                log += f'{Code.destCode2mnemonic[self._dest(command)]}={Code.compCode2mnemonic[self._comp(command)]};{Code.jumpCode2mnemonic[self._jump(command)]}'
            if 'assembly_linenos' in self.machine:
                log += f", assembly_lineno: {self.machine['assembly_linenos'][self.machine['PC']]}"
                
            print(f"D: {self.machine['D']}, A: {self.machine['A']},", 
                  'M[16:20]:', self.machine['RAM'][16:20])
            print(log)
        
        self.machine['PC'] += 1
        
        ## instruction decoding
        if self._isA(command): # A command: @symbol
            address = command[1:]
            ## instruction execution
            self.machine['A'] = bin2dec(address)
        else: # C command: dest=comp;jump
            d, c, j = self._dest(command), self._comp(command), self._jump(command)
            oldA = self.machine['A']

            ## instruction execution
            # comp: page 67
            o = Code.compFun(c)(self.machine['A'], self.machine['RAM'][oldA], self.machine['D'])

            # dest: page 68
            if d[0] == '1':
                self.machine['A'] = o
            if d[1] == '1':
                self.machine['D'] = o
            if d[2] == '1':
                self.machine['RAM'][oldA] = o
                
            # jump: page 69
            if j[0] == '1' and o < 0:
                self.machine['PC'] = oldA
            if j[1] == '1' and o == 0:
                self.machine['PC'] = oldA
            if j[2] == '1' and o > 0:
                self.machine['PC'] = oldA
        return True
            
    def __call__(self, machine_code:str, assembly_linenos: List[int]=None):
        self.load(machine_code, assembly_linenos)
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

    def __repr__(self):
        return f'HackMachine(\n{pretty_format_dict(self.machine)}\n)'


class Assembler:

    def __init__(self, free_address=16):
        self.free_address = free_address
    
    def load(self, assembly_code:str, first_pass:bool):
        self.code = assembly_code
        fstream = StringIO(self.code)
        self.parser = Parser(fstream)

        if first_pass:
            self.symbol_table = SymbolTable()        

            # first pass of code: populate symbol table with (xxx)
            # needed b/c @xxx may refer to labeled symbol
            while True:
                ok, _, _ = self.advance(first_pass=True)
                if not ok: break
        
    def advance(self, first_pass)->Tuple[bool, str, int]:
        'get to the next command, return ok, machine_code, assembly_code_lineno'
        # if first pass, only handle (xxx) for labels b/c @xxx may refer to later label
        ok, command, ass_lineno = self.parser.advance()
        if not ok:
            return False, "no more input", ass_lineno
        ct = self.parser.commandType(command)
        # decode command
        if ct == CommandType.C_COMMAND:
            dest, comp, jump = self.parser.dest(command), self.parser.comp(command), self.parser.jump(command)
            # execute / write command
            return True, f'111{Code.comp[comp]}{Code.dest[dest]}{Code.jump[jump]}', ass_lineno
        else:
            # execute / write command
            symbol = self.parser.symbol(command)
            if ct == CommandType.A_COMMAND:
                if isInt(symbol): # e.g., @128
                    address = int(symbol)
                else: # @i
                    address = 0
                    if self.symbol_table.contains(symbol):
                        address = self.symbol_table.getAddress(symbol)
                    else: # allocate new memory on second pass b/c @i can refer to later (symbol)
                        if not first_pass:
                            address = self.free_address
                            self.free_address += 1
                            self.symbol_table.addEntry(symbol, address)

                return True, f'0{dec2bin(address, 15)}', ass_lineno
            else: # (symbol)
                if first_pass:
                    self.symbol_table.addEntry(symbol, self.parser.machine_code_lineno)
                return self.advance(first_pass=first_pass)

    def __call__(self, assembly_code:str)->(str, List[int]):
        '''given ass code, return (machine code, corresponding ass_code_linenos)'''
        self.load(assembly_code, first_pass=True)
                
        # second pass
        self.load(assembly_code, first_pass=False)
        codes = []
        ass_linenos = []
        while True:
            ok, code, ass_lineno = self.advance(first_pass=False)
            if not ok: break
            codes.append(code)
            ass_linenos.append(ass_lineno)
        return '\n'.join(codes), ass_linenos

    def __repr__(self):
        return f'Assembler(free_address={self.free_address}, symbol_table=\n{self.symbol_table}\n)'

### assembly program examples
class AssemblyExamples:
    simple_memory = '''
// A=2, D=1, M[2]=1
@2
D=1
M=D
(LOOP)
    '''
    sumfive = '''
// sum 0 to 5
@i
M=0 // i = 0
@sum
M=0 // sum = 0
@5
D=A
@niter
M=D // niter = 5
(LOOP)
    // while niter - i > 0
    // jump to end if niter - i <= 0
    @niter
    D=M
    @i
    D=D-M // D=niter-i
    @END
    D;JLE
    @i
    D=M
    @sum
    M=D+M // sum+=i
    @i
    M=M+1 // i+=1
    @LOOP
    0;JMP // jump back to loop reguardless
(END)
    '''
    mult_nat = '''
// mult_nat(a, b) = b + mult_nat(a-1, B)
// a = 3
@3
D=A
@a
M=D
// b = 4
@4
D=A
@b
M=D
// output = 0
@0
D=A
@output
M=D
(LOOP)
    // return if a == 0
    @a
    D=M
    @END
    D;JEQ
    // a -= 1
    @a
    M=M-1
    // output += b
    @b
    D=M
    @output
    M=D+M
    @LOOP
    0;JMP
(END)
    '''

if __name__ == '__main__':
    assembly_example = AssemblyExamples.mult_nat # sumfive #simple_memory
    print('Assembly code')
    print('='*30)
    print_with_lines(assembly_example)
    print('='*30)
    
    
    ### translate assembly to machine code
    ass = Assembler()
    machine_codes, ass_linenos = ass(assembly_example)
    
    print('Machine code')
    print('='*30)
    print_with_lines(machine_codes)
    print('='*30)
    
    ### execute machine code
    machine = Machine(verbose=True)
    machine(machine_codes, ass_linenos)
    
    print('Machine execution')
    print('='*30)
    print(machine)
    print('='*30)
    print(ass)

