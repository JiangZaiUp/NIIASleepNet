

import os
import numpy as np
import argparse
import glob
import math
import ntpath
import shutil
import urllib
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from mne.io import concatenate_raws, read_raw_edf
import xml.etree.ElementTree as ET
import sys
from scipy import signal
import torch
import torch.nn as nn

EPOCH_SEC_SIZE = 30
FS = 250  
SIGNAL_LENGTH = EPOCH_SEC_SIZE * FS  
LATENT_DIM = 100  


class GeneratorEEG(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 128 * (SIGNAL_LENGTH // 16)),
            nn.BatchNorm1d(128 * (SIGNAL_LENGTH // 16)),
            nn.LeakyReLU(0.2)
        )
        
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.Tanh()
        )
        
        self.freq_block = nn.Sequential(
            nn.Linear(SIGNAL_LENGTH, SIGNAL_LENGTH // 2 + 1),
            nn.BatchNorm1d(SIGNAL_LENGTH // 2 + 1),
            nn.LeakyReLU(0.2),
            nn.Linear(SIGNAL_LENGTH // 2 + 1, SIGNAL_LENGTH)
        )
        
        self.output = nn.Linear(SIGNAL_LENGTH, SIGNAL_LENGTH, bias=False)
        with torch.no_grad():
            self.output.weight.fill_(3.0)
    
    def forward(self, z):
        x = self.init_fc(z)
        x = x.view(-1, 128, SIGNAL_LENGTH // 16)
        x = self.conv_blocks(x)
        x = x.squeeze(1)
        
        
        x_fft = torch.fft.fft(x)
        x_fft_abs = torch.abs(x_fft)
        
        if x_fft_abs.size(1) < SIGNAL_LENGTH:
            padding = SIGNAL_LENGTH - x_fft_abs.size(1)
            x_fft_abs = torch.nn.functional.pad(x_fft_abs, (0, padding))
        elif x_fft_abs.size(1) > SIGNAL_LENGTH:
            x_fft_abs = x_fft_abs[:, :SIGNAL_LENGTH]
        
        x_freq = self.freq_block(x_fft_abs)
        x = torch.fft.ifft(x_freq).real
        x = x - x.mean(dim=1, keepdim=True)
        output = self.output(x)
        return output

class GeneratorEOG(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 128 * (SIGNAL_LENGTH // 16)),
            nn.BatchNorm1d(128 * (SIGNAL_LENGTH // 16)),
            nn.LeakyReLU(0.2)
        )
        
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=4, padding=2),
            nn.Tanh()
        )
        
        self.freq_block = nn.Sequential(
            nn.Linear(SIGNAL_LENGTH, SIGNAL_LENGTH // 4 + 1),
            nn.BatchNorm1d(SIGNAL_LENGTH // 4 + 1),
            nn.LeakyReLU(0.2),
            nn.Linear(SIGNAL_LENGTH // 4 + 1, SIGNAL_LENGTH)
        )
        
        self.output = nn.Linear(SIGNAL_LENGTH, SIGNAL_LENGTH, bias=False)
        with torch.no_grad():
            self.output.weight.fill_(3.0)
    
    def forward(self, z):
        x = self.init_fc(z)
        x = x.view(-1, 128, SIGNAL_LENGTH // 16)
        x = self.conv_blocks(x)
        x = x.squeeze(1)
        
        
        x_fft = torch.fft.fft(x)
        x_fft_abs = torch.abs(x_fft)
        
        if x_fft_abs.size(1) < SIGNAL_LENGTH:
            padding = SIGNAL_LENGTH - x_fft_abs.size(1)
            x_fft_abs = torch.nn.functional.pad(x_fft_abs, (0, padding))
        elif x_fft_abs.size(1) > SIGNAL_LENGTH:
            x_fft_abs = x_fft_abs[:, :SIGNAL_LENGTH]
        
        x_freq = self.freq_block(x_fft_abs)
        x = torch.fft.ifft(x_freq).real
        x = x - x.mean(dim=1, keepdim=True)
        output = self.output(x)
        return output

def convert_label(lbl):
    if 1 <= lbl <= 5:
        return lbl - 1
    return None

def random_scaling(signal, scale_range=(0.95, 1.05)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale

def random_shift(signal, shift_range=(-0.05, 0.05)):
    shift = np.random.uniform(shift_range[0], shift_range[1]) * np.std(signal)
    return signal + shift

def add_gaussian_noise(signal, noise_level=0.003):
    noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
    return signal + noise

def time_warp(signal, warp_factor=0.02):
    n = len(signal)
    warp_points = int(n * warp_factor)
    start = np.random.randint(0, n - warp_points)
    end = start + warp_points
    cropped_signal = np.concatenate([signal[:start], signal[end:]])
    xp = np.linspace(0, 1, len(cropped_signal))
    warped = np.interp(np.linspace(0, 1, n), xp, cropped_signal)
    return warped

def main():
    try:
        get_ipython()
        class Args:
            data_dir = r"xxx"
            ann_dir = r"xxx"
            output_dir = r"xxx"
            select_ch = ["EEG(sec)", "EOG(L)", "EEG"]
            gan_model_dir = r"xxx"

        args = Args()
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", type=str, default=r"xxx",
                            help="File path to the PSG files.")
        parser.add_argument("--ann_dir", type=str, default=r"xxx",
                            help="File path to the annotation files.")
        parser.add_argument("--output_dir", type=str, default=r"xxx",
                            help="Directory where to save numpy files outputs.")
        parser.add_argument("--select_ch", type=str, nargs='+', default=r"xxx",
                            help="The selected channels")
        parser.add_argument("--gan_model_dir", type=str, default=r"xxx",
                            help="Directory containing pre-trained GAN models")
        args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generators = {}
    print(f"Loading GAN models from: {args.gan_model_dir}")
    
    for ch in args.select_ch:
        model_path = os.path.join(args.gan_model_dir, f"wgangp_generator_{ch}_stage1.pt")
        if os.path.exists(model_path):
            
            if "EEG" in ch:
                generator = GeneratorEEG().to(device)
                print(f"Loaded EEG generator for channel: {ch}")
            elif "EOG" in ch:
                generator = GeneratorEOG().to(device)
                print(f"Loaded EOG generator for channel: {ch}")
            else:
                print(f"Warning: Unsupported channel type {ch}, using EEG generator")
                generator = GeneratorEEG().to(device)
                
            generator.load_state_dict(torch.load(model_path, map_location=device))
            generator.eval()
            generators[ch] = generator
        else:
            print(f"Warning: GAN model not found for channel {ch} at {model_path}")
            generators[ch] = None

    ids = pd.read_csv(r"xxx", header=None, names=['a'])
    ids = ids['a'].values.tolist()

    edf_fnames = [os.path.join(args.data_dir, i + ".edf") for i in ids]
    ann_fnames = [os.path.join(args.ann_dir, i.split('-')[-1] + ".csv") for i in ids]

    edf_fnames.sort()
    ann_fnames.sort()
    edf_fnames = np.asarray(edf_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for file_id in range(len(edf_fnames)):
        output_file_path = os.path.join(args.output_dir, os.path.basename(edf_fnames[file_id])[:-4] + ".npz")
        if os.path.exists(output_file_path):
            continue
        print(edf_fnames[file_id])

        raw = read_raw_edf(edf_fnames[file_id], preload=False, stim_channel=None, verbose=None)
        raw.load_data()
        sampling_rate = raw.info['sfreq']

        raw_ch_dfs = {}
        required_channels = []
        lower_ch_names = [ch.lower() for ch in raw.info["ch_names"]]
        
        for ch_type in args.select_ch:
            try:
                lower_ch_type = ch_type.lower()
                if lower_ch_type == "eeg":
                    match_indices = [i for i, ch in enumerate(lower_ch_names) if ch.strip("()") == lower_ch_type]
                else:
                    match_indices = [i for i, ch in enumerate(lower_ch_names) if lower_ch_type in ch]
                if match_indices:
                    select_ch = raw.info["ch_names"][match_indices[0]]
                    required_channels.append(select_ch)
                    print(f"in {edf_fnames[file_id]} find channel {ch_type},channel name: {select_ch}")
                else:
                    raise IndexError
            except IndexError:
                print(f"Channel {ch_type} not found in {edf_fnames[file_id]}, available channels: {raw.info['ch_names']}, skipping this channel.")

        if required_channels:
            raw_df = raw.to_data_frame(picks=required_channels)
            for ch in required_channels:
                for ch_type in args.select_ch:
                    if ch_type.lower() in ch.lower():
                        raw_ch_dfs[ch_type] = raw_df[ch]
                        break
        else:
            print(f"No valid channels found in {edf_fnames[file_id]}, available channels: {raw.info['ch_names']}, skipping this file.")
            continue

        labels = []
        faulty_File = 0
        try:
            df = pd.read_csv(ann_fnames[file_id])
            for lbl in df['Stage']:
                try:
                    lbl = int(lbl)
                    converted_lbl = convert_label(lbl)
                    if converted_lbl is not None:
                        labels.append(converted_lbl)
                    else:
                        print(f"in {ann_fnames[file_id]}find Invalid label {lbl}")
                        faulty_File = 1
                        break
                except ValueError:
                    print(f"Error converting label {lbl} to integer in file {ann_fnames[file_id]}")
                    faulty_File = 1
                    break
        except FileNotFoundError:
            print(f"File not found: {ann_fnames[file_id]}, skipping this file.")
            continue
        except Exception as e:
            print(f"An error occurred while reading file {ann_fnames[file_id]}: {e}, skipping this file.")
            continue

        if faulty_File == 1:
            print("============================== Faulty file ==================")
            continue

        labels = np.asarray(labels)

        x_dict = {}
        n_epochs = None
        for ch_type, ch_df in raw_ch_dfs.items():
            ch_raw = ch_df.values
            if n_epochs is None:
                if len(ch_raw) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
                    raise Exception("Something wrong")
                n_epochs = len(ch_raw) / (EPOCH_SEC_SIZE * sampling_rate)
            ch_x = np.asarray(np.split(ch_raw, n_epochs)).astype(np.float32)
            x_dict[f"x_{ch_type}"] = ch_x

        y = labels.astype(np.int32)

        for ch_type, ch_x in x_dict.items():
            print(f"channel {ch_type} length: {ch_x.shape[0]}")
        print(f"label y length: {y.shape[0]}")
        
        
        stage_indices = {stage: np.where(y == stage)[0] for stage in range(5)}
        class_counts = [len(indices) for indices in stage_indices.values()]
        max_count = max(class_counts)
        
        augmented_x_dict = {key: [] for key in x_dict}
        augmented_y = []
        t = np.linspace(0, EPOCH_SEC_SIZE, len(next(iter(x_dict.values()))[0]))

        
        enhance_config = {
            
            0: {'ratio': 0.15, 'alpha': 0.12, 'eog_noise': 0.012, 'freq': 9.5},
            
            
            1: {'ratio': 0.35},  
            
            
            2: {'ratio': 0.12, 'spindle': 0.12, 'spindle_freq': 14.0, 'delta': 0.06},
            
            
            3: {'ratio': 0.20, 'delta': 0.45, 'delta_freq': 0.8, 'slow_delta_amp': 0.25},  
            
            
            4: {'ratio': 0.30, 'rapid_eye': 0.75, 'eeg_suppress': 0.80}  
        }

        for stage, indices in stage_indices.items():
            if len(indices) == 0:
                continue
            
            
            valid_indices = []
            for idx in indices:
                sig = x_dict['x_EEG(sec)'][idx]
                std = np.std(sig)
                mean_amp = np.mean(np.abs(sig))
                
                
                if stage == 0:  
                    if 10 < std < 55 and mean_amp > 10:  
                        valid_indices.append(idx)
                elif stage == 1:  
                    if 4 < std < 50 and 3 < mean_amp < 40:  
                        valid_indices.append(idx)
                elif stage == 3:  
                    if 10 < std < 70 and mean_amp > 12:  
                        valid_indices.append(idx)
                else:  
                    if 7 < std < 60 and 5 < mean_amp < 50:  
                        valid_indices.append(idx)

            
            if not valid_indices:
                valid_indices = indices
            
            
            target_count = int(max_count * 1.15)  
            augment_count = max(0, target_count - len(indices))
            augment_count = int(augment_count)

            if augment_count > 0:
                
                if stage == 1 and all(g is not None for g in generators.values()):
                    
                    three_eighth_original = int(len(indices) * 0.8)  
                    actual_generate_count = min(augment_count, three_eighth_original)
                    
                    if actual_generate_count > 0:
                        print(f"GAN generate {actual_generate_count} N1{len(indices)}）")
                    
                    for _ in range(actual_generate_count):
                        
                        for ch_type in args.select_ch:
                            
                            noise = torch.randn(1, LATENT_DIM).to(device)
                            
                            
                            with torch.no_grad():
                                gen_signal = generators[ch_type](noise).cpu().numpy().flatten()
                            
                            
                            real_idx = np.random.choice(valid_indices)
                            real_sample = x_dict[f'x_{ch_type}'][real_idx]
                            std_real = np.std(real_sample)
                            
                            
                            gen_signal = gen_signal * std_real
                            
                            
                            gen_signal = random_scaling(gen_signal, scale_range=(0.97, 1.03))
                            gen_signal = random_shift(gen_signal, shift_range=(-0.015, 0.015))
                            gen_signal = add_gaussian_noise(gen_signal, noise_level=0.003)
                            if np.random.rand() < 0.4:
                                gen_signal = time_warp(gen_signal, warp_factor=0.03)
                            
                            
                            augmented_x_dict[f'x_{ch_type}'].append(gen_signal)
                        
                        
                        augmented_y.append(1)
                
                
                elif stage != 1:  
                    selected_indices = np.random.choice(valid_indices, size=min(augment_count, len(valid_indices)), replace=False)
                    
                    for idx in selected_indices:
                        for key in x_dict:
                            orig = x_dict[key][idx]
                            aug = orig.copy()
                            
                            
                            if stage == 0:
                                alpha = enhance_config[0]['alpha'] * np.sin(
                                    2*np.pi*enhance_config[0]['freq']*t) * np.std(aug)
                                aug += alpha.astype(np.float32)
                                if 'EOG' in key:
                                    aug += enhance_config[0]['eog_noise'] * np.random.normal(
                                        0, np.std(aug), len(aug))
                            
                            
                            elif stage == 2:
                                spindle = enhance_config[2]['spindle'] * np.sin(
                                    2*np.pi*enhance_config[2]['spindle_freq']*t) * np.exp(-0.1*(t-15)**2) * np.std(aug)
                                aug += spindle.astype(np.float32)
                            
                            
                            elif stage == 3:
                                
                                delta = enhance_config[3]['delta'] * np.sin(
                                    2*np.pi*enhance_config[3]['delta_freq']*t) * np.std(aug)
                                
                                
                                if np.random.rand() < 0.8:  
                                    delta += 0.15 * np.sin(
                                        2*np.pi*(enhance_config[3]['delta_freq']/2)*t) * np.std(aug)
                                
                                
                                slow_delta = enhance_config[3]['slow_delta_amp'] * np.sin(
                                    2*np.pi*0.5*t) * np.std(aug)
                                delta += slow_delta.astype(np.float32)
                                
                                aug += delta.astype(np.float32)
                            
                            
                            elif stage == 4:
                                if 'EOG' in key:
                                    
                                    rapid_eye = enhance_config[4]['rapid_eye'] * np.sign(
                                        np.sin(2*np.pi*1.5*t)) * np.std(aug)
                                    
                                    
                                    if np.random.rand() < 0.7:
                                        rapid_eye += 0.35 * np.sign(  
                                            np.sin(2*np.pi*0.8*t)) * np.std(aug)
                                    
                                    
                                    if np.random.rand() < 0.6:
                                        rapid_eye += 0.25 * np.sign(
                                            np.sin(2*np.pi*3.0*t)) * np.std(aug)
                                    
                                    aug += rapid_eye.astype(np.float32)
                                else:
                                    
                                    aug *= enhance_config[4]['eeg_suppress']
                            
                            
                            aug = random_scaling(aug, scale_range=(0.97, 1.03))
                            aug = random_shift(aug, shift_range=(-0.015, 0.015))
                            aug = add_gaussian_noise(aug, noise_level=0.003)
                            if np.random.rand() < 0.4:
                                aug = time_warp(aug, warp_factor=0.03)
                            
                            augmented_x_dict[key].append(aug)
                        augmented_y.append(stage)

        
        for key in x_dict:
            if augmented_x_dict[key]:
                x_dict[key] = np.concatenate([x_dict[key], np.stack(augmented_x_dict[key])])
        y = np.concatenate([y, augmented_y])

        for ch_type in x_dict:
            if len(x_dict[ch_type]) != len(y):
                print(f"edf {edf_fnames[file_id]} channel {ch_type} Not corresponding")

        else:
            filename = os.path.basename(edf_fnames[file_id]).replace(".edf", ".npz")
            save_dict = {
                **x_dict,
                "y": y,
                "fs": sampling_rate
            }
            np.savez(os.path.join(args.output_dir, filename), **save_dict)
            saved_file = np.load(os.path.join(args.output_dir, filename))
            print(f" {filename} : {list(saved_file.keys())}")
            saved_file.close()
            print(" ---------- Done this file ---------")

if __name__ == "__main__":
    main()