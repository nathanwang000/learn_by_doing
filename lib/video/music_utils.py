import tqdm
import numpy as np
import imageio
import os
import math
import copy
import functools
import itertools
import tempfile

from mingus.core import chords
from mingus.containers import Bar, Note, NoteContainer, Track, Composition
from mingus.midi import fluidsynth # see https://stackoverflow.com/questions/62478717/importerrorcouldnt-find-the-fluidsynth-library how to fix import error
from mingus.containers import instrument
from mingus.containers.instrument import Instrument, Piano, Guitar
from mingus.extra import lilypond
import sys
# get current path
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
from utils import AudioFileClip
from typing import Union, Optional


def mid2wav(fn, rewrite=False):
  fn = '.'.join(fn.split('.')[:-1])
  if not os.path.exists(fn + '.wav') or rewrite:
    if os.system(f'timidity "{fn}.mid" -Ow -o "{fn}.wav"') != 0:
      os.system(f'fluidsynth -ni sound_fonts/campbellspianobeta2.sf2 "{fn}.mid" -F "{fn}.wav" -r 44100')
  else:
    print(f'file {fn}.wav already exists, skipping')
  return f'{fn}.wav'

###### overwrite from mingus.midi import midi_file_out to fix the channel writing issue
from mingus.midi import midi_file_out
from mingus.midi.midi_track import MidiTrack
from binascii import a2b_hex

# solve the issue of not setting channel correctly for midi with mingus
# with https://bspaans.github.io/python-mingus/_modules/mingus/midi/midi_track.html#MidiTrack.get_midi_data
class MyMidiTrack:
    def play_Note(self, note):
        """Convert a Note object to a midi event and adds it to the
        track_data.
    
        To set the channel on which to play this note, set Note.channel, the
        same goes for Note.velocity.
        """
        velocity = 64 # 1-2
        channel = 1
        if hasattr(note, 'dynamics'):
            if 'velocity' in note.dynamics:
                velocity = note.dynamics['velocity']
            if 'channel' in note.dynamics:
                channel = note.dynamics['channel']
        if hasattr(note, 'channel'):
            channel = note.channel
        if hasattr(note, 'velocity'):
            velocity = note.velocity
        ##### change start
        if hasattr(self, 'channel'):
            channel = self.channel
        if hasattr(self, 'velocity'):
            velocity = self.velocity
        ##### change end
        if self.change_instrument:
            self.set_instrument(channel, self.instrument)
            self.change_instrument = False
    
        self.track_data += self.note_on(channel, int(note) + 12, velocity)

    def stop_Note(self, note):
        """Add a note_off event for note to event_track."""
        velocity = 64
        channel = 1
        if hasattr(note, 'dynamics'):
            if 'velocity' in note.dynamics:
                velocity = note.dynamics['velocity']
            if 'channel' in note.dynamics:
                channel = note.dynamics['channel']
        if hasattr(note, 'channel'):
            channel = note.channel
        if hasattr(note, 'velocity'):
            velocity = note.velocity            
        ##### change start
        if hasattr(self, 'channel'):
            channel = self.channel
        if hasattr(self, 'velocity'):
            velocity = self.velocity
        ##### change end

        self.track_data += self.note_off(channel, int(note) + 12, velocity)    
    
    def play_Track(self, track):
        """Convert a Track object to MIDI events and write them to the
        track_data."""
        if hasattr(track, 'name'):
            self.set_track_name(track.name)
        self.delay = 0
        instr = track.instrument
        if hasattr(instr, 'instrument_nr'):
            print('instrument number', instr.instrument_nr)
            self.change_instrument = True
            self.instrument = instr.instrument_nr
        ##### change start
        if hasattr(instr, 'channel'):
            self.channel = instr.channel
            print('channel', instr.channel)
        if hasattr(instr, 'velocity'):
            self.velocity = instr.velocity
        ##### change end
        for bar in track:
            self.play_Bar(bar)

MidiTrack.play_Note = MyMidiTrack.play_Note
MidiTrack.stop_Note = MyMidiTrack.stop_Note
MidiTrack.play_Track = MyMidiTrack.play_Track
##### overwrite done

##### improve mingus
fluidsynth.init("sound_fonts/campbellspianobeta2.sf2")

Bar.length = float('inf') # disable Bar length check
Bar.set_meter = lambda self, *args, **kwargs: None # set to dummy

def get_audio(self, bpm=120):
  with tempfile.NamedTemporaryFile(suffix='.mid') as f:
    test_mid = f.name
    self.save(test_mid, bpm=bpm)
    test_wav = mid2wav(test_mid)
  return AudioFileClip(test_wav)

def wrap_play(item):
    def _f(self, *args, **kwargs):
        getattr(fluidsynth, 
                f'play_{item.__name__}')(self, *args, **kwargs) 
    return _f

def wrap_save(item):
    def _f(self, fn, *args, **kwargs):
        getattr(midi_file_out, 
                f'write_{item.__name__}')(fn, self, *args, **kwargs) 
    return _f

def wrap_lilypond(item):
    def _f(self, *args, **kwargs)->str:
        return getattr(lilypond, 
                    f'from_{item.__name__}')(self, *args, **kwargs) 
    return _f

def chain(f):
    @functools.wraps(f)
    def _f(self, *args, **kwargs):
        f(self, *args, **kwargs)
        return self
    return _f
  
# enhance mingus's usability
for item in [Bar, NoteContainer, Track, Composition]:
    item.play = wrap_play(item)
    item.save = wrap_save(item)
    item.get_audio = get_audio
    item.lilypond = wrap_lilypond(item)
    item.pdf = lambda self, fn: lilypond.to_pdf(self.lilypond(), fn)
    item.__add__ = chain(getattr(item, '__add__'))


##### done alias definition

def note_add(self:Note, other:Union[str,int]):
    if type(other) != str:
        other = str(other)
    o = intervals.from_shorthand(self.name, other)
    assert o, f"invalid note addition {self} + {other}"
    octave = self.octave
    if Note(o, octave) < self:
        octave += 1
    return Note(o, octave)

def note_minus(self:Note, other:Union[str,int]):
    if type(other) != str:
        other = str(other)
    o = intervals.from_shorthand(self.name, other, False)
    assert o, f"invalid note addition {self} + {other}"
    octave = self.octave
    if Note(o, octave) > self:
        octave -= 1
    return Note(o, octave)

Note.__add__ = note_add
Note.__sub__ = note_minus

def track_mul(self, other):
    '''
    create a new track object combining both tracks
    '''
    assert type(self) is Track, "self must be track"

    t = Track(self.instrument)
    if type(other) is Track:
        for beat, duration, notes in itertools.chain(self.get_notes(), other.get_notes()):
            t.add_notes(notes, duration=duration)
    else:
        raise NotImplementedError(f'{type(other)}')
    return t
    

def track_add_notes(self, note, duration=None):
    """Add a Note, note as string or NoteContainer to the last Bar.
    If note is None, treat as rest

    If the Bar is full, a new one will automatically be created.

    If the Bar is not full but the note can't fit in, this method will
    return False. True otherwise.

    An InstrumentRangeError exception will be raised if an Instrument is
    attached to the Track, but the note turns out not to be within the
    range of the Instrument.
    """

    # changed: check None for rest (only changed this one line)
    if note is not None and self.instrument is not None:
        if not self.instrument.can_play_notes(note):
            raise Exception("Note '%s' is not in range of the instrument (%s)" % (note,
                    self.instrument))
    if duration == None:
        duration = 4

    # Check whether the last bar is full, if so create a new bar and add the
    # note there
    if len(self.bars) == 0:
        self.bars.append(Bar())
    last_bar = self.bars[-1]
    if last_bar.is_full():
        self.bars.append(Bar(last_bar.key, last_bar.meter))
        # warning should hold note if it doesn't fit

    return self.bars[-1].place_notes(note, duration)
    
Track.__mul__ = track_mul
Track.add_notes = track_add_notes
