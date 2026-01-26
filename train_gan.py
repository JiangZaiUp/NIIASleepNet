
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
import torch
if torch.cuda.is_available():
    torch.cuda.init()
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import argparse
from scipy import signal
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings 


EPOCH_SEC_SIZE = 30
FS = 250
SIGNAL_LENGTH = EPOCH_SEC_SIZE * FS  
LATENT_DIM = 100
GP_WEIGHT = 5
CRITIC_ITERATIONS = 3
BATCH_SIZE = 64
TARGET_STAGE = 1
REGULARIZATION_LAMBDA = 10
torch.manual_seed(42)
np.random.seed(42)

def setup_logging(log_file):
    
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


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


class Discriminator(nn.Module):
    def __init__(self, is_eeg=True):
        super().__init__()
        
        kernel_size = 5 if is_eeg else 9
        
        
        self.conv1_output_size = self._compute_conv_output_size(SIGNAL_LENGTH, kernel_size, 2, kernel_size//2)
        self.conv2_output_size = self._compute_conv_output_size(self.conv1_output_size, kernel_size, 2, kernel_size//2)
        self.conv3_output_size = self._compute_conv_output_size(self.conv2_output_size, kernel_size, 2, kernel_size//2)
        
        self.conv_blocks = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(1, 16, kernel_size=kernel_size, stride=2, padding=kernel_size//2)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(16, 32, kernel_size=kernel_size, stride=2, padding=kernel_size//2)),
            nn.LayerNorm([32, self.conv2_output_size]),  
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(32, 64, kernel_size=kernel_size, stride=2, padding=kernel_size//2)),
            nn.LayerNorm([64, self.conv3_output_size]),  
            nn.LeakyReLU(0.2),
        )
        
        
        self.attention = nn.Sequential(
            nn.Linear(64, 1),  
            nn.Sigmoid()
        )
        
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(64 * self.conv3_output_size, 1))
        )
        
        
        self.final_conv_output_size = self.conv3_output_size
    
    def _compute_conv_output_size(self, input_size, kernel_size, stride, padding):
        
        return (input_size + 2 * padding - kernel_size) // stride + 1
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)  
        
        
        
        x_permuted = x.permute(0, 2, 1)  
        attn_weights = self.attention(x_permuted)  
        attn_weights = attn_weights.permute(0, 2, 1)  
        x = x * attn_weights
        
        x = x.view(-1, 64 * self.final_conv_output_size)
        return self.fc(x)

class R3GAN_Trainer:
    def __init__(self, stage, channel, logger, training_samples_dir):
        self.stage = stage
        self.channel = channel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.real_samples = None
        self.real_samples_stats = None
        self.training_samples_dir = training_samples_dir
        
        if str(self.device) == 'cuda':
            torch.cuda.init()
            torch.backends.cudnn.benchmark = True
        
        
        self.is_eeg = channel.lower().startswith('eeg')
        self.generator = GeneratorEEG().to(self.device) if self.is_eeg else GeneratorEOG().to(self.device)
        self.discriminator = Discriminator(is_eeg=self.is_eeg).to(self.device)
        
        
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), 
            lr=0.0001, 
            betas=(0.0, 0.9))
            
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0004,
            betas=(0.0, 0.9))
        
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=30, gamma=0.5)
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=30, gamma=0.5)
        
        self.losses_G = []
        self.losses_D = []
        self.gradient_penalties = []
        self.spectral_losses = []
        self.time_losses = []  
        logger.info(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters())}")
        logger.info(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters())}")

    def _compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def _spectral_loss(self, real, fake):
        real_fft = torch.fft.fft(real)
        fake_fft = torch.fft.fft(fake)
        return F.l1_loss(real_fft.abs(), fake_fft.abs())
    
    
    def _time_domain_loss(self, real, fake):
        return F.l1_loss(real, fake)

    def _compute_relative_loss(self, real, fake):
        real_logits = self.discriminator(real)
        fake_logits = self.discriminator(fake.detach())
        loss_D = F.softplus(fake_logits - real_logits).mean()
        loss_G = F.softplus(real_logits - fake_logits).mean()
        return loss_D, loss_G
    
    def _r1_regularization(self, real):
        real.requires_grad_(True)
        real_logits = self.discriminator(real)
        gradients = torch.autograd.grad(
            outputs=real_logits.sum(),
            inputs=real,
            create_graph=True
        )[0]
        return gradients.pow(2).sum(dim=1).mean()

    def train(self, real_samples, epochs=100):
        self.real_samples = torch.FloatTensor(real_samples).to(self.device)
        dataset = TensorDataset(self.real_samples)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        real_samples_zero_mean = real_samples - np.mean(real_samples, axis=1, keepdims=True)
        self.real_samples_stats = {
            'std': np.std(real_samples_zero_mean, axis=1, keepdims=True),
            'mean': np.zeros_like(np.mean(real_samples, axis=1, keepdims=True))
        }
        
        real_samples = real_samples_zero_mean / (self.real_samples_stats['std'] + 1e-8)
        real_samples = np.clip(real_samples, -3, 3)
        self.real_samples = torch.FloatTensor(real_samples).to(self.device)
        
        for epoch in range(epochs):
            epoch_loss_G = 0.0
            epoch_loss_D = 0.0
            epoch_gp = 0.0
            epoch_spec_loss = 0.0
            epoch_time_loss = 0.0  
            n_d_steps = 0
            n_g_steps = 0
            
            for i, (real_data,) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                
                
                self.optimizer_D.zero_grad()
                noise = torch.randn(real_data.size(0), LATENT_DIM).to(self.device)
                fake_data = self.generator(noise).detach()
                
                loss_D, _ = self._compute_relative_loss(real_data, fake_data)
                r1_penalty = self._r1_regularization(real_data) * REGULARIZATION_LAMBDA
                gradient_penalty = self._compute_gradient_penalty(real_data, fake_data) * GP_WEIGHT
                total_loss_D = loss_D + r1_penalty + gradient_penalty
                total_loss_D.backward()
                self.optimizer_D.step()
                
                epoch_loss_D += loss_D.item()
                epoch_gp += gradient_penalty.item()
                n_d_steps += 1
                
                
                if i % CRITIC_ITERATIONS == 0:
                    self.optimizer_G.zero_grad()
                    fake_data = self.generator(noise)
                    _, loss_G = self._compute_relative_loss(real_data, fake_data)
                    
                    
                    spectral_loss = self._spectral_loss(real_data, fake_data)
                    spectral_weight = min(0.2, 0.01 + epoch / epochs * 0.2)
                    
                    
                    time_loss = self._time_domain_loss(real_data, fake_data)
                    time_weight = 0.5
                    
                    
                    loss_G += spectral_weight * spectral_loss + time_weight * time_loss
                    
                    loss_G.backward()
                    self.optimizer_G.step()
                    
                    epoch_loss_G += loss_G.item()
                    epoch_spec_loss += spectral_loss.item()
                    epoch_time_loss += time_loss.item()  
                    n_g_steps += 1
            
            
            avg_loss_D = epoch_loss_D / n_d_steps if n_d_steps > 0 else 0
            avg_loss_G = epoch_loss_G / n_g_steps if n_g_steps > 0 else 0
            avg_gp = epoch_gp / n_d_steps if n_d_steps > 0 else 0
            avg_spec_loss = epoch_spec_loss / n_g_steps if n_g_steps > 0 else 0
            avg_time_loss = epoch_time_loss / n_g_steps if n_g_steps > 0 else 0  
            
            self.losses_G.append(avg_loss_G)
            self.spectral_losses.append(avg_spec_loss)
            self.time_losses.append(avg_time_loss)  
            self.losses_D.append(avg_loss_D)
            self.gradient_penalties.append(avg_gp)
            
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            if epoch % 10 == 0:
                logging.info(
                    f"[Epoch {epoch}/{epochs}] "
                    f"D_loss: {avg_loss_D:.4f} "
                    f"G_loss: {avg_loss_G:.4f} "
                    f"Spec_loss: {avg_spec_loss:.4f} "
                    f"Time_loss: {avg_time_loss:.4f} "  
                    f"GP: {avg_gp:.4f} "
                    f"LR_G: {self.scheduler_G.get_last_lr()[0]:.6f} "
                    f"LR_D: {self.scheduler_D.get_last_lr()[0]:.6f}"
                )
                self._visualize_samples(epoch)

    def _visualize_samples(self, epoch):
        
        original_mode = self.generator.training
        try:
            self.generator.eval()
            with torch.no_grad():
                noise = torch.randn(2, LATENT_DIM).to(self.device)
                gen_samples = self.generator(noise).cpu().numpy()
                gen_sample = gen_samples[0].flatten()
                
                real_samples = self.real_samples.cpu().numpy() if isinstance(self.real_samples, torch.Tensor) else np.array(self.real_samples)
                if len(real_samples) < 1:
                    raise ValueError(f"Not enough real samples ({len(real_samples)}) for visualization")
                    
                idx = np.random.choice(len(real_samples), size=1, replace=False)
                real_sample = real_samples[idx].flatten()
                
                real_sample = real_sample - real_sample.mean()
                gen_sample = gen_sample - gen_sample.mean()
                
                std = np.squeeze(self.real_samples_stats['std'][idx])
                mean = np.squeeze(self.real_samples_stats['mean'][idx])
                gen_sample = gen_sample * (std + 1e-8) + mean
                real_sample = real_sample * (std + 1e-8) + mean
                
                plt.figure(figsize=(15, 12))
                plt.suptitle(
                    f"Epoch {epoch} - {self.channel} Stage {self.stage}\n"
                    f"Real vs Generated Samples (Blue=Real, Red=Generated) in uV",
                    y=1.02,
                    fontsize=14
                )
                
                
                time_axis = np.linspace(0, EPOCH_SEC_SIZE, SIGNAL_LENGTH)
                plt.subplot(3, 3, 1)
                plt.plot(time_axis, real_sample, 'b-', alpha=0.8, linewidth=0.5)
                plt.title("Real Sample (30s) in uV", fontsize=10)
                plt.xlim(0, EPOCH_SEC_SIZE)
                y_min = min(np.min(real_sample), np.min(gen_sample))
                y_max = max(np.max(real_sample), np.max(gen_sample))
                y_padding = (y_max - y_min) * 0.1
                plt.ylim(y_min - y_padding, y_max + y_padding)
                plt.grid(True, alpha=0.3)
                plt.xlabel("Time (sec)")
                plt.ylabel("Amplitude (uV)")
                
                plt.subplot(3, 3, 2)
                plt.plot(time_axis, gen_sample, 'r-', alpha=0.8, linewidth=0.5)
                plt.title("Generated Sample (30s) in uV", fontsize=10)
                plt.xlim(0, EPOCH_SEC_SIZE)
                plt.ylim(y_min - y_padding, y_max + y_padding)
                plt.grid(True, alpha=0.3)
                plt.xlabel("Time (sec)")
                plt.ylabel("Amplitude (uV)")
                
                
                n_perseg = min(256, SIGNAL_LENGTH//4)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f_real, Pxx_real = signal.welch(real_sample, fs=FS, nperseg=n_perseg)
                    f_gen, Pxx_gen = signal.welch(gen_sample, fs=FS, nperseg=n_perseg)
                
                Pxx_real = np.clip(Pxx_real, a_min=1e-10, a_max=None)
                Pxx_gen = np.clip(Pxx_gen, a_min=1e-10, a_max=None)
                Pxx_min = max(min(Pxx_real.min(), Pxx_gen.min()), 1e-10)
                Pxx_max = max(Pxx_real.max(), Pxx_gen.max())
                
                plt.subplot(3, 3, 4)
                plt.semilogy(f_real, Pxx_real, 'b-', linewidth=1)
                plt.title("Real Spectrum (0-30Hz)", fontsize=10)
                plt.xlim(0, 30)
                plt.ylim(Pxx_min * 0.9, Pxx_max * 1.1)
                plt.grid(True, alpha=0.3)
                
                plt.subplot(3, 3, 5)
                plt.semilogy(f_gen, Pxx_gen, 'r-', linewidth=1)
                plt.title("Generated Spectrum (0-30Hz)", fontsize=10)
                plt.xlim(0, 30)
                plt.ylim(Pxx_min * 0.9, Pxx_max * 1.1)
                plt.grid(True, alpha=0.3)
                
                
                f_real, t_real, Sxx_real = signal.spectrogram(real_sample, fs=FS, nperseg=n_perseg)
                f_gen, t_gen, Sxx_gen = signal.spectrogram(gen_sample, fs=FS, nperseg=n_perseg)
                Sxx_real = np.clip(Sxx_real, a_min=1e-10, a_max=None)
                Sxx_gen = np.clip(Sxx_gen, a_min=1e-10, a_max=None)
                log_Sxx_real = np.log10(Sxx_real)
                log_Sxx_gen = np.log10(Sxx_gen)
                vmin = min(log_Sxx_real.min(), log_Sxx_gen.min())
                vmax = max(log_Sxx_real.max(), log_Sxx_gen.max())
                
                plt.subplot(3, 3, 7)
                plt.pcolormesh(t_real, f_real, log_Sxx_real, 
                            shading='auto', cmap='viridis', 
                            vmin=vmin, vmax=vmax)
                plt.title("Real Spectrogram", fontsize=10)
                plt.ylim(0, 30)
                plt.colorbar(label='Power (dB)')
                
                plt.subplot(3, 3, 8)
                plt.pcolormesh(t_gen, f_gen, log_Sxx_gen, 
                            shading='auto', cmap='viridis', 
                            vmin=vmin, vmax=vmax)
                plt.title("Generated Spectrogram", fontsize=10)
                plt.ylim(0, 30)
                plt.colorbar(label='Power (dB)')
                
                plt.tight_layout()
                os.makedirs(self.training_samples_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(self.training_samples_dir, f"{self.channel}_stage{self.stage}_epoch{epoch}.png"),
                    bbox_inches='tight',
                    dpi=150
                )
                plt.close()
                
        finally:
            self.generator.train(original_mode)

    def save_models(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.generator.state_dict(),
                  os.path.join(model_dir, f"wgangp_generator_{self.channel}_stage{self.stage}.pt"))
        torch.save(self.discriminator.state_dict(),
                  os.path.join(model_dir, f"wgangp_discriminator_{self.channel}_stage{self.stage}.pt"))
        
        pd.DataFrame({
            'Generator Loss': self.losses_G,
            'Discriminator Loss': self.losses_D,
            'Spectral Loss': self.spectral_losses,
            'Time Loss': self.time_losses,  
            'Gradient Penalty': self.gradient_penalties
        }).to_csv(os.path.join(model_dir, f"wgangp_training_log_{self.channel}_stage{self.stage}.csv"))

def convert_label(lbl):
    if 1 <= lbl <= 5:
        return lbl - 1
    return None

def prepare_training_data(data_dir, ann_dir, select_ch, file_list, logger):
    
    training_data = {ch: {stage: [] for stage in range(5)} for ch in select_ch}
    
    progress_bar = tqdm(file_list, desc="Processing files")
    for filename in progress_bar:
        try:
            edf_path = os.path.join(data_dir, f"{filename}.edf")
            label_filename = filename.split('-')[-1] + ".csv"
            ann_path = os.path.join(ann_dir, label_filename)
            
            if not os.path.exists(ann_path):
                logger.warning(f"Label file not found: {ann_path}, skipping {filename}")
                continue
                
            if not os.path.exists(edf_path):
                logger.warning(f"EDF file not found: {edf_path}, skipping")
                continue
                
            raw = read_raw_edf(edf_path, preload=True, stim_channel=None, verbose=False)
            sampling_rate = raw.info['sfreq']
            
            if sampling_rate != FS:
                logger.warning(f"Unexpected sampling rate {sampling_rate}Hz in {filename}, expected {FS}Hz")
                continue
            
            try:
                df = pd.read_csv(ann_path)
                labels = []
                for lbl in df['Stage']:
                    try:
                        converted = convert_label(int(lbl))
                        if converted is not None:
                            labels.append(converted)
                    except ValueError:
                        continue
                
                if not labels:
                    logger.warning(f"No valid labels in {ann_path}, skipping")
                    continue
                    
                labels = np.array(labels)
            except Exception as e:
                logger.error(f"Error reading annotation file {ann_path}: {e}")
                continue
            
            raw_ch_dfs = {}
            lower_ch_names = [ch.lower() for ch in raw.info["ch_names"]]
            
            for ch_type in select_ch:
                try:
                    lower_ch_type = ch_type.lower()
                    if lower_ch_type == "eeg":
                        match_indices = [i for i, ch in enumerate(lower_ch_names) if "eeg" in ch.lower()]
                    else:
                        match_indices = [i for i, ch in enumerate(lower_ch_names) if lower_ch_type in ch]
                    
                    if match_indices:
                        select_ch_name = raw.info["ch_names"][match_indices[0]]
                        raw_ch_dfs[ch_type] = raw.get_data(picks=[select_ch_name])[0]
                    else:
                        logger.warning(f"Channel {ch_type} not found in {filename}")
                except Exception as e:
                    logger.error(f"Error processing channel {ch_type} in {filename}: {e}")
                    continue
            
            for ch_type, ch_data in raw_ch_dfs.items():
                epoch_samples = SIGNAL_LENGTH
                total_samples = len(ch_data)
                n_epochs = total_samples // epoch_samples
                n_epochs = min(n_epochs, len(labels))
                
                if n_epochs <= 0:
                    logger.warning(f"Not enough data for epochs in {filename}, skipping")
                    continue
                
                ch_data = ch_data[:n_epochs * epoch_samples]
                epochs = np.array(np.split(ch_data, n_epochs))
                
                for stage in range(5):
                    stage_indices = np.where(labels[:n_epochs] == stage)[0]
                    if len(stage_indices) > 0:
                        training_data[ch_type][stage].extend(epochs[stage_indices])
        
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            continue
    
    valid_channels = []
    for ch in select_ch:
        total_samples = sum(len(training_data[ch][stage]) for stage in range(5))
        if total_samples > 0:
            for stage in range(5):
                if training_data[ch][stage]:
                    training_data[ch][stage] = np.array(training_data[ch][stage])
            valid_channels.append(ch)
        else:
            logger.warning(f"No valid samples found for channel {ch}")
    
    return {ch: training_data[ch] for ch in valid_channels}

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"xxx",
                      help="Directory containing the EDF files")
    parser.add_argument("--ann_dir", type=str, default=r"xxx",
                      help="Directory containing the annotation files")
    parser.add_argument("--model_dir", type=str, default=r"xxx",
                      help="Directory to save trained GAN models")
    parser.add_argument("--training_samples_dir", type=str, default=r"xxx",
                      help="Directory to save training samples")
    parser.add_argument("--select_ch", type=str, nargs='+', default=[],
                      help="Channels to process")
    parser.add_argument("--file_list", type=str, default=r"xxx",
                      help="Text file containing list of files to use for training")
    parser.add_argument("--epochs", type=int, default=200,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,  
                      help="Batch size for training")
    parser.add_argument("--log_file", type=str, default=r"xxx",
                      help="log path")
    args = parser.parse_args()
    
    logger = setup_logging(args.log_file)
    logger.info("\n" + "="*50)
    logger.info(f"Starting training with sampling rate {FS}Hz")
    logger.info(f"Command line arguments: {vars(args)}")
    
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    if not os.path.exists(args.ann_dir):
        logger.error(f"Annotation directory not found: {args.ann_dir}")
        raise FileNotFoundError(f"Annotation directory not found: {args.ann_dir}")
    
    try:
        with open(args.file_list, 'r', encoding='utf-8-sig') as f:
            file_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Successfully loaded {len(file_list)} files from file list")
    except Exception as e:
        logger.exception(f"Failed to read file list: {e}")
        raise
    
    if not file_list:
        logger.error("No files found in the file list")
        raise ValueError("No files found in the file list")
    
    logger.info("Preparing training data...")
    training_data = prepare_training_data(args.data_dir, args.ann_dir, args.select_ch, file_list, logger)
    
    if not training_data:
        logger.error("No valid training data prepared")
        raise ValueError("No valid training data prepared")
    
    logger.info("Training data summary:")
    for ch, stage_data in training_data.items():
        logger.info(f"Channel {ch}:")
        for stage, data in stage_data.items():
            if len(data) > 0:
                logger.info(f"  Stage {stage}: {len(data)} samples (shape: {data[0].shape})")
    
    os.makedirs(args.model_dir, exist_ok=True)
    logger.info(f"Models will be saved to: {args.model_dir}")
    
    target_stage = TARGET_STAGE
    for channel in training_data:
        if len(training_data[channel][target_stage]) > 0:
            logger.info(f"\nTraining {channel} stage {target_stage} with {len(training_data[channel][target_stage])} samples")
            
            samples = training_data[channel][target_stage]
            samples = samples.reshape(len(samples), -1)
            
            mean = np.mean(samples, axis=1, keepdims=True)
            std = np.std(samples, axis=1, keepdims=True)
            samples = (samples - mean) / (std + 1e-8)
            samples = np.clip(samples, -3, 3)
            
            trainer = R3GAN_Trainer(target_stage, channel, logger, args.training_samples_dir)
            trainer.train(samples, epochs=args.epochs)
            trainer.save_models(args.model_dir)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise