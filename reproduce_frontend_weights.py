"""
학습이 끝난 뒤, 저장된 checkpoint로 frontend dynamic weight 그래프를 재현하는 standalone 스크립트.

- 기존 코드(train_B.py / visualize.py)는 전혀 수정하지 않고, 헬퍼 함수만 재사용한다.
- 기본값은 "test set 전체 누적 평균"이라 첫 batch만 쓰던 학습 시점 그래프의
  class 편향 문제가 없다. (--mode first_batch 로 학습 시점 동작도 재현 가능)

실행 예시 (Monaural_SELD 폴더에서):
    python reproduce_frontend_weights.py -c my_experiment15
    python reproduce_frontend_weights.py -c my_experiment15 --mode first_batch
    python reproduce_frontend_weights.py -c my_experiment15 \
        --checkpoint ./output/my_experiment15/checkpoints/checkpoint_epoch337_step92001.pth
"""
import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # GUI 없는 환경에서도 저장되도록
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader

from my_data_loader import LmdbDataset
# experiment15 checkpoint는 my_model15 아키텍처로 학습되었으므로 15에서 import한다.
from models.my_model15 import ResnetConformer_seddoa_nopool_2023
from utils.cls_tools.visualize import _compute_stft_for_frontend, _get_prior_logs
from utils.sed_doa import process_raw_mic_input


def collect_dynamic_weights(frontend, dataloader, device, mode="full"):
    """
    test set을 돌면서 frontend의 dynamic mel/gcc weight를 모은다.

    Returns:
        mel_w: [N, Freq] numpy   (N = 사용한 전체 샘플 수)
        gcc_w: [N, Freq] numpy
        n_batches: 실제 사용한 batch 수
    """
    frontend.eval()
    mel_chunks, gcc_chunks = [], []
    n_batches = 0

    with torch.no_grad():
        for data in dataloader:
            wav = data["input"].to(device)            # [B, M, F, T]
            stft, mag = _compute_stft_for_frontend(frontend, wav)
            mel_w, gcc_w = frontend.get_weights(stft, mag)   # 각각 [B, Freq]

            mel_chunks.append(mel_w.detach().cpu().numpy())
            gcc_chunks.append(gcc_w.detach().cpu().numpy())
            n_batches += 1

            if mode == "first_batch":
                # 학습 시점(train_B.py)과 동일하게 첫 batch만 사용
                break

    mel_w = np.concatenate(mel_chunks, axis=0)
    gcc_w = np.concatenate(gcc_chunks, axis=0)
    return mel_w, gcc_w, n_batches


def plot_weights(frontend, mel_w, gcc_w, sr, save_path,
                 reduction="mean", prior_mode="current", title_suffix=""):
    """
    모아둔 weight 배열을 받아 prior 곡선과 함께 그린다.
    prior 계산식은 기존 visualize.py 와 동일하게 맞춘다.
    """
    # --- prior 곡선 (학습된 delta 반영) ---
    shared_prior_log, mel_prior_log, gcc_prior_log = _get_prior_logs(frontend, mode=prior_mode)

    s_scale = frontend.shared_scale.item() if isinstance(frontend.shared_scale, torch.Tensor) else frontend.shared_scale
    r_scale = frontend.residual_scale.item() if isinstance(frontend.residual_scale, torch.Tensor) else frontend.residual_scale

    mel_prior_weight = np.exp(s_scale * shared_prior_log + r_scale * mel_prior_log)
    gcc_prior_weight = np.exp(s_scale * shared_prior_log + r_scale * gcc_prior_log)

    freqs = np.linspace(0, sr / 2, frontend.n_freqs)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, mel_prior_weight, "--", label=f"Mel prior ({prior_mode})", linewidth=2)
    plt.plot(freqs, gcc_prior_weight, "--", label=f"GCC prior ({prior_mode})", linewidth=2)

    if reduction == "mean":
        mel_mean, gcc_mean = mel_w.mean(axis=0), gcc_w.mean(axis=0)
        mel_std, gcc_std = mel_w.std(axis=0), gcc_w.std(axis=0)

        plt.plot(freqs, mel_mean, label="Mel dynamic (mean)", color="tab:green")
        plt.plot(freqs, gcc_mean, label="GCC dynamic (mean)", color="tab:red")
        plt.fill_between(freqs, mel_mean - mel_std, mel_mean + mel_std, alpha=0.2, color="tab:green")
        plt.fill_between(freqs, gcc_mean - gcc_std, gcc_mean + gcc_std, alpha=0.2, color="tab:red")
    elif reduction == "none":
        for i in range(mel_w.shape[0]):
            plt.plot(freqs, mel_w[i], color="tab:green", alpha=0.3,
                     label="Mel dynamic" if i == 0 else None)
            plt.plot(freqs, gcc_w[i], color="tab:red", alpha=0.3,
                     label="GCC dynamic" if i == 0 else None)
    else:
        raise ValueError("reduction must be 'mean' or 'none'")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Weight")
    plt.title(f"FiLM-style frequency weights: prior vs dynamic ({prior_mode}){title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main(args_cli):
    # --- config 로드 (train_B.py와 동일한 방식) ---
    with open(os.path.join("config", f"{args_cli.config_name}.yaml"), "r") as f:
        args = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 모델 생성 + checkpoint 로드 ---
    model = ResnetConformer_seddoa_nopool_2023(
        in_channel=args["model"]["in_channel"],
        in_dim=args["model"]["in_dim"],
        out_dim=args["model"]["out_dim"],
        norm_path=args["data"]["norm_file"],
    ).to(device)

    ckpt_path = args_cli.checkpoint or args["model"]["pre-train_model"]
    print(f"[load] checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)          # 학습과 동일한 strict=True 로딩
    model.eval()

    # --- test set 구성 (train_B.py와 동일: split=[4], shuffle=False) ---
    test_split = [4]
    test_dataset = LmdbDataset(
        args["data"]["test_lmdb_dir"], test_split, normalized_features_wts_file=None,
        ignore=args["data"]["test_ignore"], segment_len=args["data"]["test_segment_len"],
        data_process_fn=process_raw_mic_input, is_train=False,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args["data"]["batch_size"], shuffle=False,
        num_workers=args["train"]["test_num_workers"], collate_fn=test_dataset.collater,
    )

    frontend = model.module.frontend if hasattr(model, "module") else model.frontend

    # --- weight 수집 ---
    mel_w, gcc_w, n_batches = collect_dynamic_weights(
        frontend, test_dataloader, device, mode=args_cli.mode
    )
    print(f"[info] mode={args_cli.mode}, batches={n_batches}, samples={mel_w.shape[0]}, "
          f"n_freqs={mel_w.shape[1]}")
    print(f"[info] mel dynamic  mean range: [{mel_w.mean(0).min():.3f}, {mel_w.mean(0).max():.3f}]")
    print(f"[info] gcc dynamic  mean range: [{gcc_w.mean(0).min():.3f}, {gcc_w.mean(0).max():.3f}]")

    # --- 저장 경로 (기존 학습 그래프를 덮어쓰지 않도록 별도 파일명 기본값) ---
    if args_cli.save_path:
        save_path = args_cli.save_path
    else:
        base = os.path.splitext(args["result"]["frontend_weight_plot_path"])[0]
        save_path = f"{base}_repro_{args_cli.mode}.png"

    plot_weights(
        frontend, mel_w, gcc_w,
        sr=args_cli.sr, save_path=save_path,
        reduction=args_cli.reduction, prior_mode=args_cli.prior_mode,
        title_suffix=f" | {args_cli.mode}, N={mel_w.shape[0]}",
    )
    print(f"[done] saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("reproduce frontend dynamic-weight plot")
    parser.add_argument("-c", "--config_name", type=str, default="my_experiment15",
                        help="config 파일 이름 (config/<name>.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 경로 override (기본: config의 pre-train_model)")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "first_batch"],
                        help="full=test set 전체 누적(편향 없음), first_batch=학습 시점과 동일")
    parser.add_argument("--reduction", type=str, default="mean", choices=["mean", "none"],
                        help="mean=평균±std 곡선, none=샘플별 곡선 전부")
    parser.add_argument("--prior_mode", type=str, default="current", choices=["current", "base"])
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--save_path", type=str, default=None,
                        help="저장 경로 override (기본: <config 경로>_repro_<mode>.png)")
    main(parser.parse_args())
