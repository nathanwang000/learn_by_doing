import json
import sys
import os
from io import StringIO
from typing import Tuple, List, Dict

WORD_SIZE = 16 # 16 width for hack machine

def binary_neg_str(b:str)->str:
    ''' 
    negate in 2's complement for str b
    e.g., '01' becomes '11'
    '''
    assert isinstance(b, str), f"input {b} is not str"
    assert sum(c not in '01' for c in b) == 0, f"input {b} is not binary"
    # flip(b) + '1'
    b_flip = [(0 if c == '1' else 1) for c in b]
    b_flip_plus_one = []
    carry = 1
    for digit in b_flip[::-1]:
        carry, plus_one = (digit + carry) // 2, (digit + carry) % 2
        b_flip_plus_one.append(plus_one)
    return ''.join(map(str, b_flip_plus_one[::-1]))    

def dec2bin_nat(dec:int, n_digits:int)->str:
    # decimal to binary, assume Nat
    assert dec >= 0, "input assume to be Nat"
    ret = []
    i = 0
    while i < n_digits and dec > 0:
        i += 1
        ret.append(dec % 2)
        dec //= 2
    ret += [0] * (n_digits - len(ret))
    return ''.join(map(str, ret[::-1]))

def dec2bin(dec:int, n_digits:int)->str:
    # decimal to binary in 2's complement notation
    if dec >= 0: return dec2bin_nat(dec, n_digits)
    return binary_neg_str(dec2bin_nat(-dec, n_digits))

def bin2dec_nat(b:str):
    # convert binary number str to dec int
    # assume the first digit to be 0 for 2's complement notation
    assert b[0] == '0', "function assume Nat for 2's complement notation"
    ret = 0
    for s in b:
        assert s in '01', f'{b} is not binary'
        ret = 2 * ret + int(s)
    return ret
    
def bin2dec(b:str)->int:
    ''' convert binary number str to dec int
    this function uses 2's complement to interpret the int
    that is if b start with 1 it is treated as negative'''
    if b[0] == '0': return bin2dec_nat(b)
    return -bin2dec_nat(binary_neg_str(b))

def isInt(s:str):
    try:
        int(s)
        return True
    except:
        return False

def print_with_lines(x:str):
    list_txt = x.split('\n')
    max_len = len(str(len(list_txt)))
    for i, l in enumerate(list_txt):
        print(f'{str(i).ljust(max_len+1)}: {l}')

def pretty_format_dict(d: Dict)->str:
    return json.dumps(d, indent=4)

def binary_add(a:int, b:int, word_size:int)->int:
    # TODO: rewrite this to really binary operation
    return a + b

def binary_neg(d:int, word_size:int)->int:
    # TODO: rewrite this to really binary operation 2's complement
    return -d

def binary_flip(d:int, word_size:int)->int:
    b:str = dec2bin(d, word_size)
    return bin2dec(''.join([('0' if i=='1' else '1') for i in b]))

def binary_and(a:int, b:int, word_size:int)->int:
    ab, bb = dec2bin(a, word_size), dec2bin(b, word_size)
    return bin2dec(''.join([('1' if i == j == '1' else '0')for i, j in zip(ab, bb)]))

def binary_or(a:int, b:int, word_size:int)->int:
    ab, bb = dec2bin(a, word_size), dec2bin(b, word_size)
    return bin2dec(''.join([('1' if (i == '1' or j == '1') else '0')for i, j in zip(ab, bb)]))
