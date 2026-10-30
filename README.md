# Frequency-Adaptive Auditory Frontend for Microphone-Domain Sound Event Localization and Detection

**Eojin Kim, Sungbin Im** | Soongsil University | IEEE MLSP 2026

[Code](https://github.com/eojinnn/2026mlsp_FAAF) | [Research Portfolio](https://github.com/eojinnn/Research-portfolio)

FAAF adaptively weights log-mel and GCC-PHAT features for microphone-domain SELD. **ResNet-Conformer and CST-Former are the two evaluation backends**; their architectures and training protocols are retained to isolate the effect of the proposed frontend.

## Reference Implementations

- **ResNet-Conformer:** Based on [Solution_on_3D_SELD](https://github.com/yxdong0320/Solution_on_3D_SELD), including its data preparation and training pipeline.
- **CST-Former:** Based on [CST-former](https://github.com/yusunnny/CST-former), maintained separately and not included in this repository.
- **Our contribution:** The frequency-adaptive frontend, rather than a new backend.

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Configure local dataset/feature paths in `utils/cls_tools/parameters.py` and run:

```bash
python utils/cls_tools/batch_feature_extraction.py
```

Then set the paths in `utils/lmdb_tools/convert_lmdb.py` and run:

```bash
python utils/lmdb_tools/convert_lmdb.py
```

## Training

For the ResNet-Conformer path, configure local paths, normalization and checkpoint settings in `config/my_experiment15.yaml`. Set `model.pre-train: False` for training from scratch.

```bash
python train_B.py -c my_experiment15
```

The frontend implementation used here is in `models/my_model15.py`. To inspect learned frontend weights from an existing compatible checkpoint:

```bash
python reproduce_frontend_weights.py -c my_experiment15 --checkpoint /path/to/checkpoint.pth
```

**Note:** This research repository contains additional experiment variants. The commands have not been independently verified as a clean reproduction of all published results. The CST-Former backend code is available from the linked upstream project.

## Citation

```bibtex
@inproceedings{kim2026faaf,
  title={Frequency-Adaptive Auditory Frontend for Microphone-Domain Sound Event Localization and Detection},
  author={Kim, Eojin and Im, Sungbin},
  booktitle={2026 IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2026}
}
```

The original [MIT license](LICENSE) from the ResNet-Conformer reference implementation is retained.
