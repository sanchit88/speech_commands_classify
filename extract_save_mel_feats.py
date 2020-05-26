#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:21:24 2020

@author: sanchit
"""

import numpy as np
import os 
import librosa
from glob import glob
import deepdish as dd

# define parameters 
data_dir = "/Users/sanchit/Documents/Projects/datasets/audio_data/speech_commands/"
n_fft = 2048
save_whole_dataset = False 

class Extract_Mel_Feats:
    """ extraction and saving of Mel Spetrogram features """
    def __init__(self, dir_path, n_fft=2048, is_save_everything=False):
        self.dir_path = data_dir
        self.n_fft = n_fft 
        self.hop_length = int(n_fft/4) 
        self.is_save_everything = is_save_everything 
        
    def _compute_mel_spectrogram(self, data, sampling_rate):
        """compute mel spectrogram of the signal"""
        mel_spectrogram = librosa.feature.melspectrogram(data, sr=sampling_rate)
        return librosa.power_to_db(abs(mel_spectrogram))
    
    def _load_cleaned_data(self):
        """extract features only of the cleaned data, i.e., number of samples in the audio are same"""
        dataset = dict() 
        directories=[d for d in os.listdir(self.dir_path) if os.path.isdir(d) or (not d.startswith("."))]
        for label, class_name in enumerate(directories):
            dataset[class_name] = list() 
            print(f"loading data for class: {class_name}")
            class_dir = os.path.join(self.dir_path, class_name) 
    
            print(f"number of files: {len(glob(class_dir + '/*.wav'))}")
            for file_path in glob(class_dir + '/*.wav'):
                audio_data, sampling_rate = librosa.load(file_path, duration=1.0)
                if len(audio_data) > sampling_rate or len(audio_data) < sampling_rate:
                    continue 
                mel_spec = self._compute_mel_spectrogram(audio_data, sampling_rate)
                # save the mel spectrogram features 
                dataset[class_name].append(mel_spec)
        return dataset
    
    def _load_whole_data(self):
        """extract features of all the data, i.e., when number of samples are not same then either do truncation (when 
            they are bigger) or padding (when they are smaller)."""
        dataset = dict() 
        directories=[d for d in os.listdir(self.dir_path) if os.path.isdir(d) or (not d.startswith("."))]
        for label, class_name in enumerate(directories):
            dataset[class_name] = list() 
            print(f"loading data for class: {class_name}")
            class_dir = os.path.join(self.dir_path, class_name) 
    
            print(f"number of files: {len(glob(class_dir + '/*.wav'))}")
            for file_path in glob(class_dir + '/*.wav'):
                audio_data, sampling_rate = librosa.load(file_path, duration=1.0)
                if len(audio_data) > sampling_rate:
                    pass 
                    # TODO: truncation of the samples 
                elif len(audio_data) < sampling_rate:
                    pass 
                    # TODO: padding of the samples 
                mel_spec = self._compute_mel_spectrogram(audio_data, sampling_rate)
                # save the mel spectrogram features 
                dataset[class_name].append(mel_spec)
        return dataset
    
    def extract_save_feats(self, path_to_save="./temp.h5"):
        if self.is_save_everything:
            dataset = self._load_whole_data()
        else:
            dataset = self._load_cleaned_data()
        # save the computed features 
        dd.io.save(data_dir + path_to_save, dataset, compression=None) 
        return dataset 


def main():
    extract_feats = Extract_Mel_Feats(dir_path=data_dir, n_fft=n_fft, is_save_everything=False)
    audio_dataset = extract_feats.extract_save_feats(path_to_save="audio_cleaned_dataset.h5")


if __name__ == '__main__':
    main()


