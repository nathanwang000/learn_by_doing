import itertools
import re
import copy
import warnings

from dataclasses import dataclass
from typing import Union, Optional, List, Tuple

# get current path
import os, sys
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
from music_utils import Bar, Note, NoteContainer, Track, Composition, instrument

@dataclass
class InstrumentConfig:
    instr_number:int = 0 # see midi instrument list
    channel:int = 0 # 9 for percussion
    volume:int = 100  # 0-127, as per the MIDI standard
    

############## jx notation files
@dataclass
class NoteTerm: # my notation is ([\+\-]?)([a-zA-Z]+)([\'\,]?)(\d+(\.\d+)?)?
    accent:str # '#' or 'b' or ''
    note:Optional[str] # note or None for rest
    octave:int # how many extra octave modification, ' for +1, , for -1
    duration:float # how long to play
    
def parse_jx_term(s):
    pattern = r'([\+\-]?)([a-zA-Z]+)([\'\,]*)(\d+(\.\d+)?)?'
    match = re.match(pattern, s)
    
    if match:
        accent, note, octave, duration = match.groups()[0], match.groups()[1], match.groups()[2], match.groups()[3]
        jx_note_dict = {
            'so': 'G',
            'la': 'A',
            'xi': 'B',
            'do': 'C',
            're': 'D',
            'mi': 'E',
            'fa': 'F',
            'r': None
        }
        return NoteTerm({'+': '#', '-': 'b', '': ''}[accent], 
                        jx_note_dict[note], 
                        4 + len(list(filter(lambda x: x=="'", octave))) - len(list(filter(lambda x: x==',', octave))),
                        float(duration if duration != None else 1),
                        )
    else:
        raise Exception(f'cannot parse "{s}" into Jx music notation')

def test_parse_jx_term():
    s = "-so'3.14"
    result = parse_jx_term(s)
    if result:
        print(result)
    else:
        print("Input string does not match the pattern.")

def parse_jx_string(s:str, 
                    config:InstrumentConfig=InstrumentConfig())->Track:
    '''
    jx notation allows comment '#'
    '''
    m = instrument.MidiInstrument()
    m.instrument_nr = config.instr_number
    m.channel = config.channel
    m.velocity = config.volume
    
    t = Track(instrument=m)
    for line in s.splitlines():
        line = line.split('#')[0] # use '#' as comment
        for term in line.split():
            n = parse_jx_term(term)
            t.add_notes(
                f'{n.note}{n.accent}-{n.octave}' if n.note else None,
                duration=4/n.duration # {1:4, 2: 2, 4:1} so 4/jx_notation
            )
    return t

######### jx notation ends


if __name__ == '__main__':
    melody = '''
    so0.7 so0.3 la so do' xi2 # comment the whole bar need to add to a whole number otherwise it won't work
    so0.7 so0.3 la so re' do'2
    so0.7 so0.3 so' mi' do' xi la
    fa'0.7 fa'0.3 mi' do' re' do'
    '''
    t = parse_jx_string(melody)
    t.get_audio().play()

