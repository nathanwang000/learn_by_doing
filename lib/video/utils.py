import os
import math
import imageio
import numpy as np

import matplotlib.pyplot as plt
import moviepy.video.fx.all as vfx

from moviepy.editor import VideoClip, AudioFileClip, TextClip, CompositeVideoClip, VideoFileClip, concatenate_videoclips, AudioClip
from moviepy.audio.fx.all import volumex
from moviepy.Clip import Clip
from moviepy.editor import concatenate_audioclips, CompositeAudioClip, ImageClip
from moviepy.video.io.bindings import mplfig_to_npimage
from PIL import Image
from typing import Union
from dataclasses import dataclass

def get_image_clip(image_fn, ideal_w=1920, ideal_h=1080, duration=24, fps=1, show=False):
    '''
    will align on image to fit ideal_w and ideal_h size
    '''
    from skimage.transform import resize
    canvas = np.zeros((ideal_h, ideal_w, 3))    
    img = imageio.imread(image_fn)
    h, w, _ = img.shape
    if h / w < ideal_h / ideal_w:
        img = resize(img, (int(h / w * ideal_w), ideal_w))
        pad = (ideal_h - img.shape[0]) // 2
        canvas[pad:pad+img.shape[0], :] = img
    else:
        img = resize(img, (ideal_h, int(w / h * ideal_h)))
        pad = (ideal_w - img.shape[1]) // 2
        canvas[:, pad:pad+img.shape[1]] = img

    if show:
        plt.imshow(canvas)
        print(img.shape, canvas.max())
        plt.show()

    return ImageClip((canvas * 255).astype(np.int8)).set_duration(duration).set_fps(fps)

def wrap_text(text, char_limit):
    def wrap_line(line):
        words = line.split()
        current_line = []
        wrapped_lines = []

        for word in words:
            if sum(len(w) for w in current_line) + len(word) + len(current_line) - 1 < char_limit:
                current_line.append(word)
            else:
                wrapped_lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            wrapped_lines.append(' '.join(current_line))

        return wrapped_lines

    # Split by original newlines
    lines = text.split('\n')

    # Wrap each line
    wrapped_lines = [wrap_line(line) for line in lines]

    # Flatten and join
    return '\n'.join(line for sublist in wrapped_lines for line in sublist)

import nltk
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

from nltk.tokenize import sent_tokenize
nltk.download('punkt')  # Downloading the Punkt Tokenizer models

def sentences_to_lines(text):
    sentences = sent_tokenize(text)
    return '\n'.join(sentences)

def text_effect(name, effect='\mathbf'):
    return " ".join(["$" + effect + "{" + k + "}$" for k in name.split()])

def get_museum_card(author:str, 
                    year:str, 
                    name:str,
                    char_limit:int=30,
                    bold_text_offset:int=5,
                    w=5, h=1.5, linewidth=6, fontsize=10)->np.array:
    fig = plt.figure(figsize=(w, h), linewidth=linewidth, edgecolor='black')
    fig.text(.5, .5, "\n\n".join(
       [
            text_effect(author, '\mathbf') + (" " if len(author+' '+year) < char_limit - bold_text_offset else "\n") + year, 
            wrap_text(name, char_limit)
    ]), va='center', ha='center', fontsize=fontsize)
    # plt.axis('off')
    import io
    io_buf = io.BytesIO()
    plt.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr

@dataclass
class CropRate:
  hmin:float=0
  hmax:float=1
  wmin:float=0
  wmax:float=1

def crop_img(img:np.array, crop_rate:CropRate, pixel:int=None)->np.array:
  '''
  crop image with crop_rate
  negative or >1 crop_rate is interpreted as padding
  '''
  h, w, _ = img.shape
  img = img[
      int(max(crop_rate.hmin*h,0)):int(min(crop_rate.hmax*h,h)),
      int(max(crop_rate.wmin*w,0)):int(min(crop_rate.wmax*w,w))
  ]

  w = img.shape[1]
  if crop_rate.hmin<0:
    img = np.vstack([img, np.zeros((int(h*(-crop_rate.hmin)), w, 3))])
  if crop_rate.hmax>1:
    img = np.vstack([np.zeros((int(h*(crop_rate.hmax-1)), w, 3)), img])

  h = img.shape[0]
  if crop_rate.wmin<0:
    img = np.hstack([np.zeros((h, int(w*(-crop_rate.wmin)), 3)), img])
  if crop_rate.wmax>1:
    img = np.hstack([img, np.zeros((h, int(w*(crop_rate.wmax-1)), 3))])

  if pixel:
    return resize(img, (pixel,pixel))[..., :3]
  return img[...,:3]

def video_clip_add(self, other):
    if isinstance(other, VideoClip):
        return CompositeVideoClip([self, other])
    elif isinstance(other, AudioClip):
        return self.set_audio(other)
    else:
        raise TypeError(f'Video can only be added to Video or Audio, not {type(other)}')

def video_clip_mul(self, other):
    if isinstance(other, VideoClip) or isinstance(other, AudioClip):
            return concatenate_videoclips([self, other])
    elif isinstance(other, float) or isinstance(other, int):
        return self.loop(duration=self.duration * other)
    else:
        raise TypeError(f'Video can only be multiplied by Video, Audio, or a scaler, not {type(other)}')

def clip_getitem(self, key):
    if isinstance(key, slice):
        start, stop, step = key.indices(math.ceil(self.duration))
        start, stop = max(min(start, stop), 0), min(max(start, stop), self.duration)
        if step > 0:
            return self.subclip(start, stop)
        else: # reverse the video
            return self.subclip(start, stop).fx(vfx.time_mirror).set_duration(stop-start)
    elif isinstance(key, tuple):
        raise NotImplementedError('Tuple as index')
    else:
        raise NotImplementedError('Indexing not implemented')

def clip_len(self):
    return self.duration

def audio_add(self, other):
    if isinstance(other, AudioClip):
        return CompositeAudioClip([self, other]).set_fps(self.fps)
    elif isinstance(other, VideoClip):
        return other + self
    else:
        raise TypeError(f'Audio can only be added to Audio or Video, not {type(other)}')

def audio_mul(self, other):
    if isinstance(other, AudioClip):
        return concatenate_audioclips([self, other])
    elif isinstance(other, VideoClip):
        return other * self
    elif isinstance(other, float) or isinstance(other, int):
        return self.audio_loop(duration=self.duration * other)
    else:
        raise TypeError(f'Audio can only be multiplied by Audio, Video, or a scalar, not {type(other)}')

VideoClip.__add__ = video_clip_add
VideoClip.__mul__ = video_clip_mul
AudioClip.__add__ = audio_add
AudioClip.__mul__ = audio_mul
Clip.__getitem__ = clip_getitem
Clip.play = lambda self, *args, **kwargs: self.ipython_display(*args, maxduration=self.duration+1, **kwargs)
VideoClip.save = lambda self, save_path, *args, **kwargs: self.write_videofile(save_path, *args,
                                                                               audio_codec='aac',  **kwargs)
AudioClip.save = lambda self, save_path, *args, **kwargs: self.write_audiofile(save_path, *args, **kwargs)
AudioClip.loop = AudioClip.audio_loop
AudioClip.fadein = AudioClip.audio_fadein
AudioClip.fadeout = AudioClip.audio_fadeout

def get_static_frame_factory(bg_img_path):
    # load background image as static video
    img = plt.imread(bg_img_path)

    def factory(fig_width):
        fig, ax = plt.subplots()
        fig.set_size_inches(fig_width, 
                            fig_width/img.shape[1]*img.shape[0]) # matches image size
        
        def make_frame(t):
            if t == 0:
                plt.imshow(img)
                plt.tight_layout(pad=0)
                plt.axis('off')
            return mplfig_to_npimage(fig)
            
        return make_frame
        
    return factory
    

def test(x):
    '''
    testing function, no real use
    '''
    print('testing', x)
    return x

def txt_audio_video(message, audio_path,
                    video_path=None,
                    frame_factory=None, 
                    fig_width=15,
                    video_duration=None,
                    audio_duration=None,
                    message_duration=None,
                    save_path=None, text_color='mediumblue',
                    font='BM-Jua',
                    audio_fadeout_duration=3.0,
                    text_position='center',
                  ):
    '''
    frame_factory: given fig_width, return makeframe: t->npimage
    either video_path or frame_factory needs to be non_empty
    '''
    # doc: https://moviepy.readthedocs.io/en/latest/ref/AudioClip.html
    print('using txt_audio_video')
    assert video_path is not None or frame_factory is not None, "either video_path or frame_factory needs to be non_empty"
    if video_path is not None and video_duration is None:
        video_duration = VideoFileClip(video_path).duration

    audio_duration = audio_duration or video_duration
    if audio_duration is None:
        audio_duration = 10
    if video_duration is None:
        video_duration = audio_duration
        
    music = AudioFileClip(audio_path)
    if audio_duration > music.duration:
        music = music.audio_loop(duration=audio_duration)

    # fade out music at the end
    audioclip = music.subclip(0, audio_duration).audio_fadeout(audio_fadeout_duration)
    
    if video_path:
        video = VideoFileClip(video_path)
        if video_duration > video.duration:
            # loop video
            video = video.loop(duration=video_duration)            
        animation = video.subclip(0, video_duration)
    else:
        make_frame = frame_factory(fig_width)
        animation = VideoClip(make_frame, duration=video_duration)

    if message:
        # add some text
        txt_clip = TextClip(message,
                            fontsize=9*fig_width,
                            color=text_color,
                            font=font)
        message_duration = message_duration or video_duration
        txt_clip = txt_clip.set_pos(text_position).set_duration(message_duration)
        # Overlay the text clip on the first video clip
        video = CompositeVideoClip([animation, txt_clip]).set_audio(audioclip)
    else:
        video = animation.set_audio(audioclip)

    if save_path is not None:
        video.write_videofile(save_path, fps=24)
    return video

if __name__ == '__main__':
    # https://zulko.github.io/moviepy/getting_started/working_with_matplotlib.html
    message = "Happy Birthday\nBaby"
    bg_img_path = "happy_birthday/happy birthday.png" 
    bg_music_path = "happy_birthday/happy-birthday-to-you-dance-20919.mp3"

    txt_audio_video(message, bg_music_path,
                    get_frame_factory(bg_img_path),
                    video_duration=1)

