#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:55:09 2020

@author: sanchit

@description: create samples from "_background_noise_" audio files (of TF Speech commands dataset)
by randomly cropping the signal of desired time duration.
"""
import sys
import random
import numpy as np
import librosa
from glob import glob

# define parameters
data_dir = "/Users/sanchit/Documents/Projects/datasets/audio_data/common_background/_background_noise_/"
dir_to_save = "/Users/sanchit/Documents/Projects/datasets/audio_data/common_background/background/"

sampling_rate = None # if None then librosa loads the audio file in its original sampling rate
num_audio_files = 300 # number of audio files to generate
is_distortion = False
time_duration = 1 # this is the length of a cropped audio signal in time (secs)

def load_audio(file_path):
    """loads a background noise audio wav file and create samples by randomly cropping it"""
    # load the audio file in its original sampling rate
    audio_data, sr = librosa.load(file_path, sr=sampling_rate)

    # get the common file name
    file_name = file_path.split("/")[-1]
    file_name = file_name.split(".wav")[0]

    # calculate number of samples in the time duration needed
    num_samples = int(sr*time_duration)

    # get the cut-off audio signals and save them
    for i in np.arange(num_audio_files):
        audio_cut_data = cut_sample(audio_data, num_samples)
        file_path=dir_to_save + file_name + "_" + str(i+1) + ".wav"
        save_sample(audio_cut_data, file_path, sr)
        print(f"generating signal {str(i)}, its length {len(audio_cut_data)} by cutting the original signal")

def cut_sample(whole_audio_data, num_samples):
    """generate a signal by cutting original audio signal of a desired length"""
    len_audio_data = len(whole_audio_data)
    if num_samples >= len_audio_data:
        raise Exception("Length of to be generated signal cannot be greater and equal to original audio signal")
        sys.exit(-1)

    # generate a random number which is used as a first index to cut off
    ind = random.randint(0, len_audio_data-num_samples)
    gen_data = whole_audio_data[ind:ind+num_samples]
    return gen_data

def save_sample(data, file_path, sr):
    librosa.output.write_wav(file_path, data, sr)

def main():
    for file_path in glob(data_dir + '/*.wav'):
        filename = file_path.split("/")[-1]
        print(f"###### processing {filename} #############")
        load_audio(file_path)


if __name__=="__main__":
    main()
