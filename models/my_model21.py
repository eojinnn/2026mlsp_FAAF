import torch
import math
import joblib
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18_nopool
from .conformer import ConformerBlock
from nnAudio.features.gammatone import Gammatonegram


layer_resnet = ['conv1', 'bn1', 'relu', 'layer1', 'layer1.0', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'maxpool1', 'layer2', 'layer2.0', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'maxpool2', 'layer3', 'layer3.0', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'maxpool3', 'layer4', 'layer4.0', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'conv5']


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GlobalScalar(nn.Module):
    def __init__(self, scaler_path):
        super().__init__()
        scaler = joblib.load(scaler_path)
        mean = torch.tensor(scaler.mean_, dtype=torch.float32).view(1, 10, 1, 64)
        std = torch.tensor(scaler.scale_, dtype=torch.float32).view(1, 10, 1, 64)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-8)


class AuditoryFrontend(nn.Module):
    """
    [revised my_model21]
    Static-anchored time-varying frequency weighting.

    핵심 아이디어:
    1) Static anchor는 model17처럼 MLP(mean(summary)) 경로로 계산한다.
       - summary를 먼저 시간 평균한 뒤 MLP에 넣기 때문에 clip-level frequency bias가 안정적이다.
    2) Time-varying branch는 frame-wise summary로부터 zero-mean temporal residual만 만든다.
       - residual의 시간 평균을 제거하므로 dynamic branch가 static anchor 자체를 덮어쓰지 못한다.
    3) mel/gcc를 서로 다른 방식으로 분리하지 않고, 둘 다 동일한
       static anchor + weak time-varying residual 구조를 사용한다.

    temporal_residual_strength:
        0.00 -> 거의 model17-style static weighting
        0.10 -> 권장 시작점. static을 보존하면서 약한 frame-wise correction만 허용
        1.00 -> dynamic residual을 강하게 허용
    """
    def __init__(
        self,
        sr=24000,
        n_bins=64,
        n_lags=64,
        hop_length=480,
        win_length=960,
        n_fft=1024,
        scaler_path=None,
        gate_hidden=16,
        eps=1e-8,
        coherence_window_ms=100.0,
        temporal_residual_strength=0.25,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_lags = n_lags
        self.n_freqs = n_fft // 2 + 1
        self.eps = eps
        self.sr = sr
        self.temporal_residual_strength = temporal_residual_strength

        # Local temporal window for frame-wise coherence.
        # With the default hop_length=480 at 24 kHz, 100 ms corresponds to 5 STFT frames.
        hop_ms = 1000.0 * hop_length / sr
        coherence_window_frames = max(1, int(round(coherence_window_ms / hop_ms)))
        if coherence_window_frames % 2 == 0:
            coherence_window_frames += 1
        self.coherence_window_ms = coherence_window_ms
        self.coherence_window_frames = coherence_window_frames

        self.init_slope = 3.0
        self.init_transition_ratio = 0.5

        # Mel filterbank.
        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_freqs,
            n_mels=n_bins,
            f_min=0.0,
            f_max=sr / 2,
            sample_rate=sr,
            norm="slaney",
        )
        self.register_buffer("mel_filters", mel_filters)
        self.register_buffer("window", torch.hann_window(win_length))

        # Prior bias in log-weight space.
        # freq_axis: low -> 0, high -> 1
        freq_axis = torch.linspace(0.0, 1.0, steps=self.n_freqs)
        prior_curve = torch.tanh(self.init_slope * (freq_axis - self.init_transition_ratio))

        # shared importance prior starts from neutral 0.
        self.register_buffer("shared_prior_log", torch.zeros(self.n_freqs))
        self.register_buffer("mel_prior_base", 0.35 * prior_curve)
        self.register_buffer("gcc_prior_base", -0.35 * prior_curve)

        self.mel_prior_delta = nn.Parameter(torch.zeros_like(prior_curve))
        self.gcc_prior_delta = nn.Parameter(torch.zeros_like(prior_curve))
        self.mel_prior_delta.requires_grad_(False)
        self.gcc_prior_delta.requires_grad_(False)
        self.prior_delta_limit = 0.15

        # Static summary network: MLP(mean(summary)).
        self.summary_mlp = nn.Sequential(
            nn.Linear(3, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, gate_hidden),
            nn.Tanh(),
        )

        # Dynamic summary network: MLP(summary_t).
        # Separate MLP를 두어 static anchor와 dynamic residual 역할을 분리한다.
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(3, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, gate_hidden),
            nn.Tanh(),
        )

        # Static FiLM modulation.
        self.shared_gamma = nn.Linear(gate_hidden, 1)
        self.shared_beta = nn.Linear(gate_hidden, 1)
        self.mel_gamma = nn.Linear(gate_hidden, 1)
        self.mel_beta = nn.Linear(gate_hidden, 1)
        self.gcc_gamma = nn.Linear(gate_hidden, 1)
        self.gcc_beta = nn.Linear(gate_hidden, 1)

        # Dynamic FiLM modulation for zero-mean temporal residuals.
        self.dyn_shared_gamma = nn.Linear(gate_hidden, 1)
        self.dyn_shared_beta = nn.Linear(gate_hidden, 1)
        self.dyn_mel_gamma = nn.Linear(gate_hidden, 1)
        self.dyn_mel_beta = nn.Linear(gate_hidden, 1)
        self.dyn_gcc_gamma = nn.Linear(gate_hidden, 1)
        self.dyn_gcc_beta = nn.Linear(gate_hidden, 1)

        # Branch-wise post scaling.
        self.mel_post_scale = nn.Parameter(torch.tensor(1.0))
        self.gcc_post_scale = nn.Parameter(torch.tensor(1.0))

        # Log-weight scale / clamp range.
        self.max_log_weight = 0.40   # exp(+/-0.4) ~= [0.67, 1.49]
        self.shared_scale = 0.20
        self.residual_scale = 0.12
        self.scaler = GlobalScalar(scaler_path)

        self._init_film_params()

    def _init_film_params(self):
        """
        All FiLM layers start from prior-preserving behavior.
        Static branch:
            prior * (1 + 0) + 0 = prior.
        Dynamic branch:
            prior is produced initially, then its temporal mean is subtracted,
            so the initial dynamic residual is exactly zero.
        """
        film_layers = [
            self.shared_gamma, self.shared_beta,
            self.mel_gamma, self.mel_beta,
            self.gcc_gamma, self.gcc_beta,
            self.dyn_shared_gamma, self.dyn_shared_beta,
            self.dyn_mel_gamma, self.dyn_mel_beta,
            self.dyn_gcc_gamma, self.dyn_gcc_beta,
        ]
        for layer in film_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def set_prior_trainable(self, flag=True):
        self.mel_prior_delta.requires_grad_(flag)
        self.gcc_prior_delta.requires_grad_(flag)

    def get_current_prior_logs(self):
        mel_prior = self.mel_prior_base + self.prior_delta_limit * torch.tanh(self.mel_prior_delta)
        gcc_prior = self.gcc_prior_base + self.prior_delta_limit * torch.tanh(self.gcc_prior_delta)
        return mel_prior, gcc_prior

    def _standardize_freqwise(self, x):
        """
        Standardize summary features.

        x: [B, Freq] or [B, Freq, Frames]
        - 2D static summary: normalize across frequency.
        - 3D dynamic summary: normalize across frequency and time per clip.
        """
        if x.dim() == 2:
            reduce_dim = -1
        elif x.dim() == 3:
            reduce_dim = (1, 2)
        else:
            raise ValueError(f"Expected [B, Freq] or [B, Freq, Frames], got {tuple(x.shape)}")

        mean = x.mean(dim=reduce_dim, keepdim=True)
        std = x.std(dim=reduce_dim, keepdim=True, unbiased=False).clamp_min(1e-5)
        return (x - mean) / std

    def _local_mean_time(self, x):
        """
        Local moving average along the STFT time axis.
        x: [B, Freq, Frames]
        return: [B, Freq, Frames]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected [B, Freq, Frames], got {tuple(x.shape)}")

        B, n_freqs, n_frames = x.shape
        kernel_size = min(self.coherence_window_frames, n_frames)
        if kernel_size % 2 == 0:
            kernel_size = max(1, kernel_size - 1)

        if kernel_size == 1:
            return x

        pad = kernel_size // 2
        x_ = x.reshape(B * n_freqs, 1, n_frames)
        x_ = F.avg_pool1d(
            x_,
            kernel_size=kernel_size,
            stride=1,
            padding=pad,
            count_include_pad=False,
        )
        return x_.reshape(B, n_freqs, n_frames)

    def _branch_norm(self, x):
        # x: [B, C, Feat, Frames]
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-5)
        return (x - mean) / std

    def _film_modulation(self, feat, prior, gamma_layer, beta_layer):
        """
        FiLM modulation of a prior in log-weight space.

        feat : [B, Freq, H] or [B, Freq, Frames, H]
        prior: [B, Freq] or [B, Freq, Frames]
        """
        gamma = torch.tanh(gamma_layer(feat)).squeeze(-1)
        beta = torch.tanh(beta_layer(feat)).squeeze(-1)
        out = prior * (1.0 + gamma) + beta
        return out

    def _compute_gate_summaries(self, stft, mag):
        """
        Compute static and dynamic summaries for cue weighting.

        stft: [B, M, Freq, Frames] complex
        mag : [B, M, Freq, Frames] real

        Return:
            summary_static : [B, Freq, 3]
                model17-style static path: mean over time first, then normalize.
            summary_dynamic: [B, Freq, Frames, 3]
                frame-wise path for zero-mean temporal residuals.
        """
        power = mag.pow(2)  # [B, M, Freq, Frames]

        # 1) Spectral summary: frame-wise log-power.
        log_power = torch.log1p(power.mean(dim=1))  # [B, Freq, Frames]

        # 2) Temporal variation summary: frame-wise spectral velocity.
        delta_power = torch.abs(power[:, :, :, 1:] - power[:, :, :, :-1])
        delta_power = F.pad(delta_power, (1, 0), mode="constant", value=0.0)
        velocity_summary = torch.log1p(delta_power.mean(dim=1))  # [B, Freq, Frames]

        # 3) Spatial summary: local coherence across microphone pairs.
        B, M, _, _ = stft.shape
        coh_list = []
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                X1 = stft[:, m1, :, :]
                X2 = stft[:, m2, :, :]

                cross = X1 * torch.conj(X2)
                cross_real = self._local_mean_time(cross.real)
                cross_imag = self._local_mean_time(cross.imag)
                cross_mean = torch.complex(cross_real, cross_imag)

                psd1 = self._local_mean_time(X1.abs().pow(2))
                psd2 = self._local_mean_time(X2.abs().pow(2))

                coh = cross_mean.abs() / (torch.sqrt(psd1 * psd2) + self.eps)
                coh = coh.clamp(0.0, 1.0)
                coh_list.append(coh)

        spatial_conf = torch.stack(coh_list, dim=1).mean(dim=1)  # [B, Freq, Frames]

        # Static path: mean over time first, then frequency-wise standardization.
        log_power_static = self._standardize_freqwise(log_power.mean(dim=-1))
        spatial_conf_static = self._standardize_freqwise(spatial_conf.mean(dim=-1))
        velocity_static = self._standardize_freqwise(velocity_summary.mean(dim=-1))
        summary_static = torch.stack(
            [log_power_static, spatial_conf_static, velocity_static], dim=-1
        )  # [B, Freq, 3]

        # Dynamic path: keep time, then per-clip standardization.
        log_power_dynamic = self._standardize_freqwise(log_power)
        spatial_conf_dynamic = self._standardize_freqwise(spatial_conf)
        velocity_dynamic = self._standardize_freqwise(velocity_summary)
        summary_dynamic = torch.stack(
            [log_power_dynamic, spatial_conf_dynamic, velocity_dynamic], dim=-1
        )  # [B, Freq, Frames, 3]

        return summary_static, summary_dynamic

    def get_weights(self, stft, mag):
        """
        Return:
            mel_weight: [B, Freq, Frames]
            gcc_weight: [B, Freq, Frames]
        """
        B, _, _, n_frames = stft.shape
        summary_static, summary_dynamic = self._compute_gate_summaries(stft, mag)

        # ---------------------------------------------------------
        # 1) Static anchor: model17-style MLP(mean(summary)).
        # ---------------------------------------------------------
        feat_static = self.summary_mlp(summary_static)  # [B, Freq, H]

        shared_prior_static = self.shared_prior_log.view(1, -1).expand(B, -1)
        mel_prior_1d, gcc_prior_1d = self.get_current_prior_logs()
        mel_prior_static = mel_prior_1d.view(1, -1).expand(B, -1)
        gcc_prior_static = gcc_prior_1d.view(1, -1).expand(B, -1)

        shared_static_log = self._film_modulation(
            feat_static, shared_prior_static, self.shared_gamma, self.shared_beta
        )
        mel_static_res_log = self._film_modulation(
            feat_static, mel_prior_static, self.mel_gamma, self.mel_beta
        )
        gcc_static_res_log = self._film_modulation(
            feat_static, gcc_prior_static, self.gcc_gamma, self.gcc_beta
        )

        mel_base_log = self.shared_scale * shared_static_log + self.residual_scale * mel_static_res_log
        gcc_base_log = self.shared_scale * shared_static_log + self.residual_scale * gcc_static_res_log

        mel_base_log = mel_base_log.unsqueeze(-1).expand(-1, -1, n_frames)
        gcc_base_log = gcc_base_log.unsqueeze(-1).expand(-1, -1, n_frames)

        # ---------------------------------------------------------
        # 2) Weak time-varying residual: zero-mean over time.
        # ---------------------------------------------------------
        feat_dynamic = self.dynamic_mlp(summary_dynamic)  # [B, Freq, Frames, H]

        shared_prior_dynamic = self.shared_prior_log.view(1, -1, 1).expand(B, -1, n_frames)
        mel_prior_dynamic = mel_prior_1d.view(1, -1, 1).expand(B, -1, n_frames)
        gcc_prior_dynamic = gcc_prior_1d.view(1, -1, 1).expand(B, -1, n_frames)

        dyn_shared_log = self._film_modulation(
            feat_dynamic, shared_prior_dynamic, self.dyn_shared_gamma, self.dyn_shared_beta
        )
        dyn_mel_res_log = self._film_modulation(
            feat_dynamic, mel_prior_dynamic, self.dyn_mel_gamma, self.dyn_mel_beta
        )
        dyn_gcc_res_log = self._film_modulation(
            feat_dynamic, gcc_prior_dynamic, self.dyn_gcc_gamma, self.dyn_gcc_beta
        )

        # Remove temporal mean so that the dynamic branch only changes frame-wise deviations.
        dyn_shared_log = dyn_shared_log - dyn_shared_log.mean(dim=-1, keepdim=True)
        dyn_mel_res_log = dyn_mel_res_log - dyn_mel_res_log.mean(dim=-1, keepdim=True)
        dyn_gcc_res_log = dyn_gcc_res_log - dyn_gcc_res_log.mean(dim=-1, keepdim=True)

        mel_dynamic_log = self.shared_scale * dyn_shared_log + self.residual_scale * dyn_mel_res_log
        gcc_dynamic_log = self.shared_scale * dyn_shared_log + self.residual_scale * dyn_gcc_res_log

        alpha = self.temporal_residual_strength
        mel_log = mel_base_log + alpha * mel_dynamic_log
        # [mel-only dynamic 실험] gcc는 dynamic residual을 no-op(LR)·LE 손해만 유발하므로 제거,
        # model17식 완전 static으로 되돌린다. (gcc_dynamic_log / dyn_gcc_* 는 미사용으로 남김.)
        # gcc_log = gcc_base_log + alpha * gcc_dynamic_log
        gcc_log = gcc_base_log

        mel_log = torch.clamp(mel_log, -self.max_log_weight, self.max_log_weight)
        gcc_log = torch.clamp(gcc_log, -self.max_log_weight, self.max_log_weight)

        mel_weight = torch.exp(mel_log)
        gcc_weight = torch.exp(gcc_log)

        return mel_weight, gcc_weight

    def forward(self, wav, return_weights=False):
        """
        wav: [B, M, F, T]
        """
        B, M, n_frames_in, T = wav.shape
        wav = wav.reshape(B * M, n_frames_in * T)

        # 1) shared STFT.
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        stft = stft.reshape(B, M, stft.shape[1], stft.shape[2])

        if stft.shape[-1] > n_frames_in:
            stft = stft[:, :, :, :n_frames_in]

        mag = torch.abs(stft)

        # 2) static-anchored time-varying FiLM weights.
        mel_weight, gcc_weight = self.get_weights(stft, mag)
        mel_weight = mel_weight.to(stft.device)
        gcc_weight = gcc_weight.to(stft.device)

        # 3) Mel branch.
        mel_w = mel_weight.unsqueeze(1)
        power = (mag ** 2) * mel_w
        mel = torch.matmul(
            power.transpose(-1, -2), self.mel_filters
        ).transpose(-1, -2)   # [B, M, n_mels, Frames]

        mel = 10.0 * torch.log10(mel.clamp_min(1e-10))
        max_val = mel.amax(dim=(-2, -1), keepdim=True)
        mel = torch.maximum(mel, max_val - 80.0)

        # Branch-wise normalization is intentionally not applied here,
        # to keep consistency with the current strong static frontend setting.
        # mel = self.mel_post_scale * self._branch_norm(mel)
        mel = self.mel_post_scale * mel

        # 4) GCC-PHAT branch.
        cc_list = []
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                X1 = stft[:, m1, :, :]
                X2 = stft[:, m2, :, :]

                R = X1 * torch.conj(X2)
                R_phat = R / (torch.abs(R) + 1e-8)

                R_weighted = R_phat * gcc_weight

                cc = torch.fft.irfft(R_weighted, n=self.n_fft, dim=1)
                cc = torch.fft.fftshift(cc, dim=1)

                center = self.n_fft // 2
                start = center - self.n_lags // 2
                end = start + self.n_lags
                cc_cropped = cc[:, start:end, :]   # [B, n_lags, Frames]
                cc_list.append(cc_cropped)

        gcc = torch.stack(cc_list, dim=1)   # [B, num_pairs, n_lags, Frames]
        # gcc = self.gcc_post_scale * self._branch_norm(gcc)
        gcc = self.gcc_post_scale * gcc

        # 5) concat.
        out = torch.cat([mel, gcc], dim=1)   # [B, M+num_pairs, Feat, Frames]
        out = out.permute(0, 1, 3, 2)        # [B, C, Time, Feat]
        out = self.scaler(out)

        if return_weights:
            mel_prior_1d, gcc_prior_1d = self.get_current_prior_logs()
            return out, {
                "mel_weight": mel_weight,
                "gcc_weight": gcc_weight,
                "mel_prior_log": mel_prior_1d.detach(),
                "gcc_prior_log": gcc_prior_1d.detach(),
                "mel_prior_delta": self.mel_prior_delta.detach(),
                "gcc_prior_delta": self.gcc_prior_delta.detach(),
                "temporal_residual_strength": self.temporal_residual_strength,
            }
        return out


class ResnetConformer_seddoa_nopool_2023(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, n_mel_bins=64, sig_len=480, fs=24000, norm_path=None):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        encoder_dim = 256
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = 8
        self.frontend = AuditoryFrontend(scaler_path=norm_path)
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim=encoder_dim,
                dim_head=32,
                heads=8,
                ff_mult=2,
                conv_expansion_factor=2,
                conv_kernel_size=7,
                attn_dropout=0.1,
                ff_dropout=0.1,
                conv_dropout=0.1
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        self.sed_out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, 13),
            nn.Sigmoid()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim),
            nn.Tanh()
        )

    def set_frontend_prior_trainable(self, flag=True):
        self.frontend.set_prior_trainable(flag)

    def forward(self, x, return_frontend_weights=False):
        B, M, T, L = x.shape
        x = self.frontend(x, return_weights=return_frontend_weights)
        frontend_info = None
        if return_frontend_weights:
            x, frontend_info = x

        conv_outputs = self.resnet(x)
        N, C, T, W = conv_outputs.shape
        conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C * W)

        conformer_outputs = self.input_projection(conv_outputs)

        for layer in self.conformer_layers:
            conformer_outputs = layer(conformer_outputs)
        outputs = conformer_outputs.permute(0, 2, 1)
        outputs = self.t_pooling(outputs)
        outputs = outputs.permute(0, 2, 1)
        sed = self.sed_out_layer(outputs)
        doa = self.out_layer(outputs)

        pred = torch.cat((sed, doa), dim=-1)
        if return_frontend_weights:
            return pred, frontend_info

        return pred
