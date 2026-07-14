"""
my_model18(time-varying frontend)의 dynamic weight를 Freq x Time heatmap으로 시각화.

- 기존 visualize.py / train_B.py 는 수정하지 않는 별도 스크립트.
- 기존 곡선(plot_frontend_dynamic_weights)은 시간축을 평균해 [Freq] 곡선만 보여주지만,
  이 스크립트는 시간축을 보존해 가중치가 "언제 / 어느 주파수에서" 강해지는지 본다.

mel/gcc 두 브랜치를 나란히 그리며, 색은 1.0(=prior 유지)을 기준으로 한 발산형 컬러맵.
  > 1.0 : 해당 (주파수, 시점) 강조
  < 1.0 : 억제

실행 예시 (Monaural_SELD 폴더에서):
    python visualize_frontend_heatmap.py -c my_experiment18
    python visualize_frontend_heatmap.py -c my_experiment18 --sample 3
    python visualize_frontend_heatmap.py -c my_experiment18 --batch_mean
"""
import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import yaml
from torch.utils.data import DataLoader

from my_data_loader import LmdbDataset
from models.my_model18 import ResnetConformer_seddoa_nopool_2023
from utils.cls_tools.visualize import _compute_stft_for_frontend
from utils.sed_doa import process_raw_mic_input


def get_time_varying_weights(frontend, wav, device):
    """
    wav: [B, M, F, T]
    returns mel_w, gcc_w : [B, Freq, Frames] numpy
    """
    frontend.eval()
    with torch.no_grad():
        stft, mag = _compute_stft_for_frontend(frontend, wav.to(device))
        mel_w, gcc_w = frontend.get_weights(stft, mag)   # [B, Freq, Frames]
    return mel_w.detach().cpu().numpy(), gcc_w.detach().cpu().numpy()


def _prior_weights(frontend):
    """designed prior에 해당하는 [Freq] mel/gcc weight (FiLM gamma=beta=0일 때 값)."""
    shared = frontend.shared_prior_log.detach().cpu().numpy()
    mel_p, gcc_p = frontend.get_current_prior_logs()
    mel_p = mel_p.detach().cpu().numpy()
    gcc_p = gcc_p.detach().cpu().numpy()
    s = frontend.shared_scale.item() if isinstance(frontend.shared_scale, torch.Tensor) else frontend.shared_scale
    r = frontend.residual_scale.item() if isinstance(frontend.residual_scale, torch.Tensor) else frontend.residual_scale
    mel_pw = np.exp(s * shared + r * mel_p)
    gcc_pw = np.exp(s * shared + r * gcc_p)
    return mel_pw, gcc_pw


def _imshow_weight(ax, w2d, freqs, dur_s, title, center=1.0):
    """
    w2d: [Freq, Frames] heatmap 1개.
    center=1.0 : 절대 가중치(1.0=prior 기준 발산)
    center=0.0 : 편차(0 중심 대칭, prior/시간평균을 뺀 경우)
    """
    if center == 0.0:
        m = max(abs(float(w2d.min())), abs(float(w2d.max())), 1e-6)
        norm = Normalize(vmin=-m, vmax=m)            # 0 중심 대칭
    else:
        vmin = min(w2d.min(), center - 1e-3)
        vmax = max(w2d.max(), center + 1e-3)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    im = ax.imshow(
        w2d, origin="lower", aspect="auto", cmap="coolwarm", norm=norm,
        extent=[0.0, dur_s, freqs[0], freqs[-1]],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return im


def plot_heatmaps(mel_w2d, gcc_w2d, sr, hop_s, save_path, suptitle="",
                  center=1.0, cbar_label="weight (1.0 = prior)"):
    """mel/gcc heatmap을 나란히 저장. *_w2d: [Freq, Frames]."""
    n_freqs, n_frames = mel_w2d.shape
    freqs = np.linspace(0, sr / 2, n_freqs)
    dur_s = n_frames * hop_s

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    im0 = _imshow_weight(axes[0], mel_w2d, freqs, dur_s, "Mel dynamic weight", center=center)
    im1 = _imshow_weight(axes[1], gcc_w2d, freqs, dur_s, "GCC dynamic weight", center=center)
    fig.colorbar(im0, ax=axes[0], label=cbar_label)
    fig.colorbar(im1, ax=axes[1], label=cbar_label)
    if suptitle:
        fig.suptitle(suptitle)

    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(args_cli):
    with open(os.path.join("config", f"{args_cli.config_name}.yaml"), "r") as f:
        args = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResnetConformer_seddoa_nopool_2023(
        in_channel=args["model"]["in_channel"],
        in_dim=args["model"]["in_dim"],
        out_dim=args["model"]["out_dim"],
        norm_path=args["data"]["norm_file"],
    ).to(device)

    ckpt_path = args_cli.checkpoint or args["model"]["pre-train_model"]
    print(f"[load] checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # test set (train_B.py와 동일: split=[4], shuffle=False)
    test_dataset = LmdbDataset(
        args["data"]["test_lmdb_dir"], [4], normalized_features_wts_file=None,
        ignore=args["data"]["test_ignore"], segment_len=args["data"]["test_segment_len"],
        data_process_fn=process_raw_mic_input, is_train=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args["data"]["batch_size"], shuffle=False,
        num_workers=args["train"]["test_num_workers"], collate_fn=test_dataset.collater,
    )

    frontend = model.module.frontend if hasattr(model, "module") else model.frontend
    hop_s = frontend.hop_length / args_cli.sr

    # 첫 batch 가져오기 (shuffle=False라 결정적)
    data = next(iter(test_dataloader))
    wav = data["input"].to(device)
    wav_names = data.get("wav_names", None)

    mel_w, gcc_w, = get_time_varying_weights(frontend, wav, device)  # [B, Freq, Frames]
    print(f"[info] weights shape: mel={mel_w.shape}, gcc={gcc_w.shape} (B, Freq, Frames)")

    # --- 시각화 대상 선택: 특정 sample vs batch 평균 ---
    if args_cli.batch_mean:
        mel_2d, gcc_2d = mel_w.mean(axis=0), gcc_w.mean(axis=0)
        tag = "batch_mean"
        suptitle = f"{args_cli.config_name} | time-varying frontend weights (batch mean, N={mel_w.shape[0]})"
    else:
        idx = args_cli.sample
        mel_2d, gcc_2d = mel_w[idx], gcc_w[idx]
        name = wav_names[idx] if wav_names is not None else f"idx{idx}"
        tag = f"sample{idx}"
        suptitle = f"{args_cli.config_name} | time-varying frontend weights ({name})"

    # --- prior(또는 시간평균) 제거: 편차만 보기 ---
    center, cbar_label, dev_tag = 1.0, "weight (1.0 = prior)", ""
    if args_cli.deviation == "prior":
        mel_pw, gcc_pw = _prior_weights(frontend)          # [Freq]
        mel_2d = mel_2d - mel_pw[:, None]                  # prior 빼기 (시간축 broadcast)
        gcc_2d = gcc_2d - gcc_pw[:, None]
        center, cbar_label, dev_tag = 0.0, "weight − prior", "_dev-prior"
        suptitle += "  [deviation from prior]"
    elif args_cli.deviation == "timemean":
        mel_2d = mel_2d - mel_2d.mean(axis=1, keepdims=True)  # 순수 시간변동만
        gcc_2d = gcc_2d - gcc_2d.mean(axis=1, keepdims=True)
        center, cbar_label, dev_tag = 0.0, "weight − time mean", "_dev-timemean"
        suptitle += "  [deviation from time-mean]"

    if args_cli.deviation != "none":
        print(f"[info] deviation={args_cli.deviation}, "
              f"mel dev range=[{mel_2d.min():.3f}, {mel_2d.max():.3f}], "
              f"gcc dev range=[{gcc_2d.min():.3f}, {gcc_2d.max():.3f}]")

    # 저장 경로 (기존 frontend_weights*.png 안 덮어쓰도록 별도 파일명)
    if args_cli.save_path:
        save_path = args_cli.save_path
    else:
        base = os.path.splitext(args["result"]["frontend_weight_plot_path"])[0]
        save_path = f"{base}_heatmap_{tag}{dev_tag}.png"

    plot_heatmaps(mel_2d, gcc_2d, sr=args_cli.sr, hop_s=hop_s,
                  save_path=save_path, suptitle=suptitle,
                  center=center, cbar_label=cbar_label)
    print(f"[done] saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("time-varying frontend weight heatmap")
    parser.add_argument("-c", "--config_name", type=str, default="my_experiment18")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 경로 override (기본: config의 pre-train_model)")
    parser.add_argument("--sample", type=int, default=0,
                        help="batch 내 시각화할 sample index (기본 0)")
    parser.add_argument("--batch_mean", action="store_true",
                        help="특정 sample 대신 batch 전체 평균 heatmap")
    parser.add_argument("--deviation", type=str, default="prior",
                        choices=["none", "prior", "timemean"],
                        help="none=절대값, prior=designed prior 빼기(기본), timemean=시간평균 빼기(순수 시간변동)")
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--save_path", type=str, default=None,
                        help="저장 경로 override (기본: <config 경로>_heatmap_<tag>.png)")
    main(parser.parse_args())
