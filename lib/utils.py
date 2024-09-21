import json
import sys
import os
from io import StringIO
from typing import Tuple, List, Dict

WORD_SIZE = 16 # 16 width for hack machine

def dec2bin(dec:int, n_digits:int)->str:
    # decimal to binary, assume positive
    ret = []
    i = 0
    while i < n_digits and dec > 0:
        i += 1
        ret.append(dec % 2)
        dec //= 2
    ret += [0] * (n_digits - len(ret))
    return ''.join(map(str, ret[::-1]))

def bin2dec(b:str)->int:
    # convert binary number str to dec int
    ret = 0
    for s in b:
        assert s in '01', f'{b} is not binary'
        ret = 2 * ret + int(s)
    return ret

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
