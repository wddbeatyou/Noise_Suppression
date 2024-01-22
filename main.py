# -*- coding: utf-8 -*-
# @Author: wangdongdong
# @Time: 2024/1/15 9:16
import copy
import glob
import os
import re
import time
import tkinter
import wave
from tkinter import *
import tkinter.messagebox
import pyaudio
# import winsound
import pygame
import pygame.mixer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from Speech_noise_reduction_model import Noise_Suppression_Model
from tensorflow.keras import backend
import tensorflow as tf

# 初始化混音器
pygame.mixer.init()

class denoWindow:
    def __init__(self, win, ww, wh):
        self.play_recording = []
        self.timestamp = None
        self.gettimestamp = None
        self.after_noise_path = "./data/after_denoising/"
        self.recording_path = "./data/recording/"
        self.mkdir()
        self.mode = Noise_Suppression_Model()
        _, self.encoder, self.session_list, self.reset_op = self.mode.rnn_encoder()
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.title("语音降噪模块")
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))
        self.img_src_path = None

        self.textlabe = Label(text="语音降噪功能", fg="white", bg='black', width=12, height=2, font=("Helvetica", 20))
        self.textlabe.place(x=420, y=35)

        self.button1 = Button(self.win, text='开始录音', width=10, height=2, command=self.start)
        self.button1.place(x=300, y=450)

        self.button2 = Button(self.win, text='降噪处理', width=10, height=2, command=self.denoise)
        self.button2.place(x=417, y=450)

        self.button3 = Button(self.win, text='播放原音频', width=10, height=2, command=self.play)
        self.button3.place(x=534, y=450)

        self.button4 = Button(self.win, text='播放降噪音频', width=10, height=2, command=self.playno)
        self.button4.place(x=650, y=450)

    def mkdir(self):
        if not os.path.exists(self.after_noise_path):
            os.mkdir(self.after_noise_path)
        if not os.path.exists(self.recording_path):
            os.mkdir(self.recording_path)
        return

    def start(self):
        record_second = 5
        chunk = 1024
        fmat = pyaudio.paInt16
        channels = 1
        rate = 16000
        p = pyaudio.PyAudio()
        stream = p.open(format=fmat,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
        self.timestamp = int(time.time())
        self.gettimestamp = copy.deepcopy(self.timestamp)
        wave_out_path = f"./data/recording/{self.gettimestamp}.wav"
        if wave_out_path not in self.play_recording:
            self.play_recording.append(wave_out_path)
        wf = wave.open(wave_out_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(fmat))
        wf.setframerate(rate)

        for i in tqdm(range(0, int(rate / chunk * record_second))):
            data = stream.read(chunk)
            wf.writeframes(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
        tkinter.messagebox.showinfo('提示', '处理成功')

    def denoise(self):
        print(self.play_recording)
        if not self.play_recording:
            tkinter.messagebox.showinfo('提示', '语音录制失败，请重新录音')
        filename = self.play_recording[-1]
        if not re.search(str(self.gettimestamp), filename):
            tkinter.messagebox.showinfo('提示', '语音录制失败，请重新录音')
        # mode = Noise_Suppression_Model()
        # _, encoder, session_list, reset_op = mode.rnn_encoder()
        sess = backend.get_session()
        sess.run(tf.global_variables_initializer())

        self.encoder.load_weights('./train_mode/model_DPCRN_SNR+logMSE_causal_sinw.h5')
        self.mode.test(filename, model=self.encoder, sess=sess, reset=True, reset_op=self.reset_op, session_list=self.session_list)

        tkinter.messagebox.showinfo('提示', '处理成功')

    def play(self):
        if not self.play_recording:
            tkinter.messagebox.showinfo('提示', '语音录制失败，请重新录音')
        filename = self.play_recording[-1]
        # winsound.PlaySound(filename, winsound.SND_FILENAME)
        # 加载声音文件
        sound = pygame.mixer.Sound(filename)

        # 播放声音
        sound.play()
        tkinter.messagebox.showinfo('提示', '音频播放结束')

    def playno(self):
        # after_noise = "./data/after_denoising/"
        after_noise_list = glob.glob(os.path.join(self.after_noise_path, "*.wav"))
        filename_list = [i for i in after_noise_list if re.search(str(self.gettimestamp), i)]
        if not filename_list:
            tkinter.messagebox.showinfo('提示', '语音降噪失败，请重新降噪')
        filename = filename_list[0]
        # winsound.PlaySound(filename, winsound.SND_FILENAME)
        # 加载声音文件
        sound = pygame.mixer.Sound(filename)

        # 播放声音
        sound.play()
        tkinter.messagebox.showinfo('提示', '音频播放结束')

if __name__ == '__main__':
    win = Tk()
    ww = 1000
    wh = 600
    img_gif = tkinter.PhotoImage(file="2.gif")
    label_img = tkinter.Label(win, image=img_gif, width="998", height="600")
    label_img.place(x=0, y=0)
    denoWindow(win, ww, wh)
    screenWidth, screenHeight = win.maxsize()
    geometryParam = '%dx%d+%d+%d' % (ww, wh, (screenWidth - ww) / 2, (screenHeight - wh) / 2)
    win.geometry(geometryParam)
    win.mainloop()

