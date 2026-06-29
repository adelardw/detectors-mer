# multimodal-deepfake-recognition rppg + fau

Мультимодальный детектор дипфейков на основе физиологических (rPPG) и мимических (FAU) признаков.

[English](#english) | [Русский](#русский)

---

## Quickstart — inference (`run.py`)

`run.py` takes a checkpoint and a path, runs the model, and writes one JSON object per video to a JSONL file.

### Environment setup

Install with **uv** (recommended):

```bash
uv sync
```

…or with a plain **virtualenv + pip**:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

Make sure a `.env` file exists (set `EXPERIMENTS_CFG_FOLDER` if you use a custom config folder).

> **Backbone weights are NOT required for inference from a trained checkpoint.** A trained `.ckpt` already contains all weights (swin + ME-GraphAU + DeepFakesON-Phys), so the swin/ME-GraphAU `.pth` files are loaded only if present and the result is identical without them. You only need `bash env.sh` (below) to download backbone weights for **training from scratch**.

```bash
bash env.sh   # only for training from scratch
```

> The commands below are shown with `uv run`. If you installed into an **activated virtualenv** instead, just drop the `uv run` prefix and call `python` directly — e.g. `python run.py ...` instead of `uv run run.py ...`.

**Label mapping (from the training folder names, sorted alphabetically): `fake = 0`, `real = 1`.**

Run on a folder of videos (scanned recursively):

```bash
uv run run.py \
    -ckpt path/to/checkpoint.ckpt \
    -c src/experiments/base_config.yml \
    -d /path/to/videos \
    -o predictions.jsonl
```

Run on a single video file (just point `-d` at the file):

```bash
uv run run.py \
    -ckpt path/to/checkpoint.ckpt \
    -d /path/to/clip.mp4 \
    -o prediction.jsonl
```

The same via plain `python` (after `source .venv/bin/activate`):

```bash
python run.py \
    -ckpt path/to/checkpoint.ckpt \
    -d /path/to/clip.mp4 \
    -o prediction.jsonl
```

Options: `-c` config (default `src/experiments/base_config.yml`), `-bs` batch size (default 8), `-nw` workers (default 4), `--no_face_detector` to disable the MTCNN face crop, `--threshold/-t` to decide by `P(fake) ≥ t` instead of argmax (default: argmax ≈ 0.5). CUDA is used automatically if available, otherwise CPU.

### Web demo — player + JSON

A small local web UI: drop in a video, watch it in the player, get the JSON verdict. No extra dependencies (stdlib server); the model is loaded once at startup and reuses the exact `run.py` pipeline.

```bash
uv run webapp/server.py            # or, in an activated venv: python webapp/server.py
```

Then open <http://127.0.0.1:8000> in a browser. By default it uses the delivered DF checkpoint (`experimental_results/exp_120626/best-epoch=08-val_auc=0.8485.ckpt`). Options: `-ckpt` checkpoint, `-c` config, `-p` port (default 8000), `--host`, `--no_face_detector`.

Output — one JSON object per line:

```json
{"video": "clip_001.mp4", "label": "fake", "label_id": 0, "prob": 0.97, "probs": {"fake": 0.97, "real": 0.03}}
```

`video` is the path relative to `-d`, `label`/`label_id` is the predicted class, `prob` is its probability, `probs` holds the full per-class distribution. Frame sampling is deterministic (centered clip), so the same input always yields the same prediction.

## Быстрый старт — инференс (`run.py`)

`run.py` принимает чекпоинт и путь, прогоняет модель и пишет по одному JSON-объекту на видео в JSONL-файл.

### Настройка окружения

Установка через **uv** (рекомендуется):

```bash
uv sync
```

…или через обычный **virtualenv + pip**:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

Убедитесь, что существует файл `.env` (укажите `EXPERIMENTS_CFG_FOLDER`, если используете свою папку с конфигами).

> **Веса backbone НЕ нужны для инференса по обученному чекпоинту.** Обученный `.ckpt` уже содержит все веса (swin + ME-GraphAU + DeepFakesON-Phys), поэтому файлы swin/ME-GraphAU `.pth` загружаются только при наличии, а без них результат идентичен. `bash env.sh` (ниже) нужен только для скачивания весов backbone под **обучение с нуля**.

```bash
bash env.sh   # только для обучения с нуля
```

> Команды ниже показаны с `uv run`. Если вы установили зависимости в **активированный virtualenv**, просто уберите префикс `uv run` и вызывайте `python` напрямую — например, `python run.py ...` вместо `uv run run.py ...`.

**Маппинг меток (из имён папок обучения, отсортированных по алфавиту): `fake = 0`, `real = 1`.**

Запуск на папке с видео (сканируется рекурсивно):

```bash
uv run run.py \
    -ckpt path/to/checkpoint.ckpt \
    -c src/experiments/base_config.yml \
    -d /path/to/videos \
    -o predictions.jsonl
```

Запуск на одном видеофайле (просто укажите файл в `-d`):

```bash
uv run run.py \
    -ckpt path/to/checkpoint.ckpt \
    -d /path/to/clip.mp4 \
    -o prediction.jsonl
```

То же самое через обычный `python` (после `source .venv/bin/activate`):

```bash
python run.py \
    -ckpt path/to/checkpoint.ckpt \
    -d /path/to/clip.mp4 \
    -o prediction.jsonl
```

Опции: `-c` конфиг (по умолчанию `src/experiments/base_config.yml`), `-bs` размер батча (по умолчанию 8), `-nw` воркеры (по умолчанию 4), `--no_face_detector` — отключить кроп лица MTCNN, `--threshold/-t` — решать по `P(fake) ≥ t` вместо argmax (по умолчанию argmax ≈ 0.5). CUDA используется автоматически при наличии, иначе CPU.

### Веб-демо — плеер + JSON

Небольшой локальный веб-интерфейс: закидываешь видео, смотришь его в плеере, получаешь вердикт в JSON. Без дополнительных зависимостей (сервер на stdlib); модель грузится один раз при старте и переиспользует тот же пайплайн, что и `run.py`.

```bash
uv run webapp/server.py            # или, в активированном venv: python webapp/server.py
```

Затем открой <http://127.0.0.1:8000> в браузере. По умолчанию используется отдаваемый DF-чекпоинт (`experimental_results/exp_120626/best-epoch=08-val_auc=0.8485.ckpt`). Опции: `-ckpt` чекпоинт, `-c` конфиг, `-p` порт (по умолчанию 8000), `--host`, `--no_face_detector`.

Формат вывода — по одному JSON-объекту на строку:

```json
{"video": "clip_001.mp4", "label": "fake", "label_id": 0, "prob": 0.97, "probs": {"fake": 0.97, "real": 0.03}}
```

`video` — путь относительно `-d`, `label`/`label_id` — предсказанный класс, `prob` — его вероятность, `probs` — полное распределение по классам. Сэмплирование кадров детерминированное (клип из центра ролика), поэтому один и тот же вход всегда даёт одно и то же предсказание.

---

## English

Research repository for **multimodal deepfake detection**. The model jointly leverages:

- **FAU branch** — frame-level facial action unit features (Swin Transformer Tiny + MEGraphAU GNN)
- **rPPG branch** — video-level physiological features (PhysNet, blood volume pulse signal)
- **Q-Former fusion** — Transformer Decoder with 32 learnable queries cross-attends to both modalities
- **Attention Pooler + MLP head** — binary classification (REAL / FAKE)
- **Optional multi-task heads** — auxiliary classifiers for gender, ethnicity, and emotion

## Architecture

![MDF architecture](docs/architecture.png)

See [docs/architecture.md](docs/architecture.md) for a detailed component description.

The model processes a video clip `[B, 3, T, H, W]` through two parallel branches:

1. **FAU branch (frame-level).** Each frame is passed through the frozen (or fine-tuned) MEGraphAU encoder (Swin-T backbone + GNN). The raw `[B·T, N_AU, D]` features are projected to `embed_dim`, then positional encoding is applied **per AU** — each of the 12 AUs gets its own temporal trajectory. A segment embedding `0` marks all FAU tokens.

2. **rPPG branch (video-level).** The full clip is passed through PhysNet, producing a per-frame feature sequence `[B, T, D_phys]`. Features are projected to `embed_dim`, sinusoidal positional encoding is added, and segment embedding `1` marks all rPPG tokens.

Both token sequences are concatenated and used as **memory** in a 6-layer Transformer Decoder. A set of 32 learnable query embeddings cross-attends to this memory. The decoded queries are aggregated by a 2-layer Attention Pooler (softmax-weighted sum), then passed through LayerNorm + Dropout to the binary classifier.

When multi-task heads are enabled (`num_gender/ethnicity/emotion_classes > 0`), the pooled feature also feeds into separate linear heads, and task weights are balanced via **uncertainty weighting** (Kendall et al. 2018).

### Model parameters

| Parameter | Default |
|---|---|
| FAU backbone | Swin Transformer Tiny |
| rPPG backbone | PhysNet |
| Input frames | 32 |
| Embedding dim | 512 |
| Num queries | 32 |
| Decoder layers | 6 |
| Attention heads | 8 |
| Dropout | 0.3 |
| Num AU classes | 12 |

## Datasets

Trained and evaluated on three public deepfake datasets:

| Dataset | Description |
|---|---|
| **FF++** (FaceForensics++) | Face-swapped videos, multiple manipulation methods |
| **CelebDF** (Celeb-DeepFake) | High-quality celebrity deepfake videos |
| **VCDF-X** | AI-generated face content |

Preprocessing crops faces with an MTCNN detector; the train/val/test split is a random 70/15/15 (`seed=42`). See **Results** below for the split caveat (and the grouped-split fix), and for how the current model differs from the first results.

## Results — how to read this section

This repository reports results from **two different training pipelines**. Please read this before looking at the tables.

**1. Current model — the one delivered for testing.** A deepfake-pretrained rPPG encoder (DeepFakesON-Phys) + ME-GraphAU FAU branch, trained jointly on a **mix** of FF++ + CelebDF + VCDF-X. This is the relevant model (section *Current model* below).

- Its in-domain **test-split** numbers (FF++ / CelebDF / VCDF-X) look near-perfect (0.97–1.00 AUROC) but are **optimistic because of a data-split issue**: each dataset stores augmented variants of the same clip as separate files, and a file-level split placed some of these copies in both train and test (measured: 34% of all test clips, up to ~53% on CelebDF and FF++). Treat these as an *upper bound*, not as accuracy on unseen data.
- The **honest** number comes from a fully **held-out set** (53 videos, verified by content hashing to share nothing with the training data, produced by different generators): **0.815 accuracy / 0.844 AUROC**. This is the realistic expectation on new, unseen videos.
- The split issue is already fixed for future training (grouped split); details in the *Current model* section.

**2. Legacy ablation — older pipeline, qualitative only.** Single-dataset training, cross-dataset evaluation, on an earlier training/split pipeline and the **validation** split. Keep it only as **motivation**: it shows that a model trained on a single dataset does **not** generalize to the others — which is exactly why the current model is trained on a mix. **Do not compare these numbers head-to-head with the current model** (different pipeline, different split, validation not test).

> **Bottom line for whoever evaluates the model:** expect performance around the held-out figures (**~0.82 accuracy / ~0.84 AUROC**) on genuinely unseen data. The near-1.0 in-domain numbers are inflated and should not be used as the expected accuracy.

## Legacy ablation — single-dataset cross-evaluation

> ⚠️ **First results, older pipeline — qualitative motivation only.** These are the initial cross-dataset numbers (added in commit `b04112f`, 2026-03-30). They come from an early pipeline that differs substantially from the current model — **no face-detector crop** (frames used as-is), original (non-deepfake-pretrained) backbones, fewer frames per clip, no contrastive / memory-bank loss — and they report the **validation** split (`seed=42`). Do not compare the absolute values with the current model (see *Current model* → *What changed* below). What matters here is the *pattern*, not the values: a single-dataset model collapses on the other two (the off-diagonal cells), which motivates mixed training.

Each model is trained on one dataset and evaluated on all three.

### Accuracy

| Train \ Test | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.8316** | 0.7616 | 0.5479 |
| **CelebDF** | 0.6464 | **0.9342** | 0.4944 |
| **VCDF-X** | 0.4842 | 0.4875 | **0.9269** |
| **Mix (all)** | 0.8057 | **0.9728** | **0.9131** |

### F1 (macro)

| Train \ Test | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.8622** | 0.7320 | 0.5489 |
| **CelebDF** | 0.3914 | **0.9602** | 0.2585 |
| **VCDF-X** | 0.3140 | 0.1692 | **0.9055** |
| **Mix (all)** | 0.8077 | **0.9809** | **0.9073** |

### AUROC

| Train \ Test | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.9758** | 0.8166 | 0.5768 |
| **CelebDF** | 0.7538 | **0.9999** | 0.3458 |
| **VCDF-X** | 0.4497 | 0.5445 | **0.9799** |
| **Mix (all)** | 0.9351 | **0.9981** | **0.9752** |

### Observations

- Each single-dataset model achieves strong in-domain performance (0.83–0.93 accuracy).
- Cross-domain generalization is poor for single-dataset models — VCDF-X and CelebDF artifacts differ significantly from FF++.
- **Mix training** (FF++ + CelebDF + VCDF-X) achieves the best overall generalization: 0.81 / 0.97 / 0.91 accuracy and 0.94 / 0.998 / 0.975 AUROC.

<details>
<summary>Full per-experiment metrics</summary>

**FF++ → FF++**

| Metric | Value |
|---|---:|
| Loss | 0.1478 |
| Accuracy | 0.8316 |
| F1 (macro) | 0.8622 |
| Precision | 0.9033 |
| Recall | 0.8316 |
| AUROC | 0.9758 |

Per-class: crop_img acc=0.9807, f1=0.9634 | real acc=0.6825, f1=0.7611

**FF++ → CelebDF**

| Metric | Value |
|---|---:|
| Loss | 0.3493 |
| Accuracy | 0.7616 |
| F1 (macro) | 0.7320 |
| Precision | 0.7116 |
| Recall | 0.7616 |
| AUROC | 0.8166 |

Per-class: crop_img acc=0.9049, f1=0.9238 | real acc=0.6184, f1=0.5402

**FF++ → VCDF-X**

| Metric | Value |
|---|---:|
| Loss | 1.2221 |
| Accuracy | 0.5479 |
| F1 (macro) | 0.5489 |
| Precision | 0.5540 |
| Recall | 0.5479 |
| AUROC | 0.5768 |

Per-class: fake acc=0.7828, f1=0.7551 | real acc=0.3130, f1=0.3427

**CelebDF → CelebDF**

| Metric | Value |
|---|---:|
| Loss | 0.0446 |
| Accuracy | 0.9342 |
| F1 (macro) | 0.9602 |
| Precision | 0.9908 |
| Recall | 0.9342 |
| AUROC | 0.9999 |

Per-class: crop_img acc=1.0000, f1=0.9908 | real acc=0.8684, f1=0.9296

**CelebDF → FF++**

| Metric | Value |
|---|---:|
| Loss | 3.2892 |
| Accuracy | 0.6464 |
| F1 (macro) | 0.3914 |
| Precision | 0.5987 |
| Recall | 0.6464 |
| AUROC | 0.7538 |

Per-class: crop_img acc=0.2928, f1=0.4530 | real acc=1.0000, f1=0.3298

**CelebDF → VCDF-X**

| Metric | Value |
|---|---:|
| Loss | 3.8568 |
| Accuracy | 0.4944 |
| F1 (macro) | 0.2585 |
| Precision | 0.4681 |
| Recall | 0.4944 |
| AUROC | 0.3458 |

Per-class: fake acc=0.0352, f1=0.0667 | real acc=0.9535, f1=0.4503

**VCDF-X → VCDF-X**

| Metric | Value |
|---|---:|
| Loss | 0.2047 |
| Accuracy | 0.9269 |
| F1 (macro) | 0.9055 |
| Precision | 0.8915 |
| Recall | 0.9269 |
| AUROC | 0.9799 |

Per-class: fake acc=0.9028, f1=0.9387 | real acc=0.9511, f1=0.8722

**VCDF-X → FF++**

| Metric | Value |
|---|---:|
| Loss | 2.3390 |
| Accuracy | 0.4842 |
| F1 (macro) | 0.3140 |
| Precision | 0.4896 |
| Recall | 0.4842 |
| AUROC | 0.4497 |

Per-class: crop_img acc=0.2541, f1=0.3898 | real acc=0.7143, f1=0.2381

**VCDF-X → CelebDF**

| Metric | Value |
|---|---:|
| Loss | 2.7093 |
| Accuracy | 0.4875 |
| F1 (macro) | 0.1692 |
| Precision | 0.4792 |
| Recall | 0.4875 |
| AUROC | 0.5445 |

Per-class: crop_img acc=0.0672, f1=0.1244 | real acc=0.9079, f1=0.2140

**Mix → FF++**

| Metric | Value |
|---|---:|
| Loss | 0.2222 |
| Accuracy | 0.8057 |
| F1 (macro) | 0.8077 |
| Precision | 0.8098 |
| Recall | 0.8057 |
| AUROC | 0.9351 |

Per-class: crop_img acc=0.9448, f1=0.9434 | real acc=0.6667, f1=0.6720

**Mix → CelebDF**

| Metric | Value |
|---|---:|
| Loss | 0.0368 |
| Accuracy | 0.9728 |
| F1 (macro) | 0.9809 |
| Precision | 0.9894 |
| Recall | 0.9728 |
| AUROC | 0.9981 |

Per-class: crop_img acc=0.9981, f1=0.9953 | real acc=0.9474, f1=0.9664

**Mix → VCDF-X**

| Metric | Value |
|---|---:|
| Loss | 0.1849 |
| Accuracy | 0.9131 |
| F1 (macro) | 0.9073 |
| Precision | 0.9022 |
| Recall | 0.9131 |
| AUROC | 0.9752 |

Per-class: fake acc=0.9338, f1=0.9436 | real acc=0.8924, f1=0.8711

</details>

## Current model — mixed training

**This is the model delivered for testing.** Trained jointly on a mix of VCDF-X + CelebDF + FF++; the rPPG branch uses a deepfake-pretrained encoder (DeepFakesON-Phys) and the FAU branch uses ME-GraphAU (Swin-Tiny). Config `src/experiments/base_config.yml` (`num_frames=64`, `num_queries=64`).

**What changed since the first results (commit `b04112f`).** The improvement over the legacy ablation came from the combination of:

- **Deepfake-pretrained rPPG encoder** — DeepFakesON-Phys (pretrained for deepfake detection) replaces the original PhysNet.
- **MTCNN face-detector crop** added to preprocessing (frames were previously used as-is).
- **Contrastive / metric learning with a memory bank** on the fused embedding (`metric_loss_type`, `memory_bank_size` in the config).
- **More frames per clip** — `num_frames` 32 → 64.
- **Mixed-dataset training** (FF++ + CelebDF + VCDF-X) plus other tuning.

Training (folder mode, random 70/15/15 split, seed=42):

```bash
uv run src/train.py -c src/experiments/base_config.yml \
    -d /path/to/vcdfx_videos/ \
    -d /path/to/celebdf_videos/ \
    -d /path/to/ffpp_videos/ \
    -bs 4 -nw 4
```

Evaluation on the test split (`evaluate.py -d ... -s test -od {1..3}` reproduces the training split exactly — same dataset order, seed=42 — so test indices are disjoint from train), plus a **held-out set** that was never part of training (verified: zero sha256 overlap between its 53 videos and all 32,150 training videos; different source pools and generators):

| Dataset | Split | N | Accuracy | F1 (macro) | AUROC | Loss |
|---|---|---:|:---:|:---:|:---:|:---:|
| VCDF-X | test | 2107 | 0.9752 | 0.9664 | 0.9974 | 0.0758 |
| CelebDF | test | 1590 | 0.9976 | 0.9975 | 1.0000 | 0.0248 |
| FF++ | test | 1126 | 1.0000 | 1.0000 | 1.0000 | 0.0261 |
| **Held-out set** | **held out** | 53 | **0.8150** | **0.7876** | **0.8437** | 0.5313 |

<details>
<summary>Confusion matrices & per-class metrics</summary>

**VCDF-X (test)** — fake: acc=0.9646, f1=0.9789 | real: acc=0.9859, f1=0.9539

|  | pred fake | pred real |
|---|---:|---:|
| **fake** | 1415 | 52 |
| **real** | 9 | 631 |

**CelebDF (test)** — fake: acc=1.0000, f1=0.9974 | real: acc=0.9952, f1=0.9976

|  | pred fake | pred real |
|---|---:|---:|
| **fake** | 756 | 0 |
| **real** | 4 | 830 |

**FF++ (test)** — fake: acc=1.0000, f1=1.0000 | real: acc=1.0000, f1=1.0000

|  | pred fake | pred real |
|---|---:|---:|
| **fake** | 529 | 0 |
| **real** | 0 | 597 |

**Held-out set** — fake: acc=0.7353, f1=0.8197 | real: acc=0.8947, f1=0.7556

|  | pred fake | pred real |
|---|---:|---:|
| **fake** | 25 | 9 |
| **real** | 2 | 17 |

</details>

**Caveat — train/test content overlap in test splits.** The 70/15/15 split is done at the *file* level, while the datasets store several augmented variants of the same source clip as separate files. Measured on the exact reproduced split (seed=42): **34.1% of test videos are augmented copies of train videos** — VCDF-X 10.3%, CelebDF 52.5%, FF++ 52.8%. On top of that, fakes of the same source video / actor end up in both train and test. The near-perfect FF++/CelebDF test numbers are therefore inflated; **the held-out set (0.815 acc / 0.844 AUROC) is the honest estimate of generalization**.

Fixed since 2026-06-12: `train.py`/`evaluate.py` now use a **grouped split** by default — all variants of the same source clip stay on one side of the split, and file traversal is sorted (deterministic across filesystems). Checkpoints trained before the fix must be evaluated with `--legacy_split`.

## Repository structure

```
src/
  models/
    rppg_p_fau.py           # DeepfakeDetector — main model
    rppg_p_fau_lightning.py # FauRPPGDeepFakeRecognizer — Lightning module (multi-task)
    fau_classifier.py       # FAU-only classifier
    fau_lightning.py        # FAU-only Lightning module
    rppg_classifier.py      # rPPG-only classifier
    rppg_lightning.py       # rPPG-only Lightning module
  backbones/
    fau_encoder.py          # FAUEncoder — wraps MEGraphAU (MEFARG)
    rppg_encoder.py         # RPPGEncoder — wraps PhysNet
    pos.py                  # Sinusoidal positional encoding
    av_former.py            # AVFormer utilities
    MEGraphAU/              # ME-GraphAU submodule (Swin + GNN for AU detection)
    rPPGToolbox/            # rPPG-Toolbox submodule (PhysNet and others)
  data/
    dataset.py              # VideoFolderDataset — folder-based loading
    meta_dataset.py         # MetaVideoDataset — CSV-based multi-task loading
    transforms.py           # VideoTransform — consistent frame-level augmentations
    processor.py            # FaceDetector (MTCNN) + Processor pipeline
    split.py                # experimental split utility (unused)
  pooler/
    attn_pooler.py          # AttentionPooler — softmax-weighted aggregation
    base_pooler.py
  loss/
    contrastive.py          # InfoNCEConsistencyLoss
  experiments/
    base_config.yml         # Standard config (no aux heads)
    meta_config.yml         # Multi-task config (gender=2, ethnicity=5, emotion=8)
    fau_config.yml          # FAU-only training config
    rppg_config.yml         # rPPG-only training config
  train.py                  # Main training entrypoint
  train_fau.py              # FAU-only training
  train_rppg.py             # rPPG-only training
  eval.py                   # GradCAM / feature visualization
evaluate.py                 # Evaluation entrypoint (all three dataset modes)
env.sh                      # Interactive weight downloader
load.py                     # Weight download helpers
docs/
  architecture.md           # Detailed architecture description
  architecture.drawio       # Architecture diagram source
  architecture.png          # Architecture diagram
  val_*.png                 # Validation metric plots
```

## Setup

### 1. Install dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### 2. Download pretrained weights

> Needed **only for training from scratch**. For inference/evaluation from a trained `.ckpt` you can skip this step — the checkpoint already contains all weights (the swin/ME-GraphAU `.pth` files are loaded only if present).

Interactive script to download FAU and backbone weights:

```bash
bash env.sh
```

Place FAU weights at:
```
src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth
```

Place rPPG weights (from [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)) at:
```
src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth
```

### 3. Configure environment

```bash
cp .env.example .env  # set EXPERIMENTS_CFG_FOLDER if needed
```

## Training

Training is implemented in **PyTorch Lightning** with DDP support.

### Mode 1 — folder-based datasets

Dataset structure: `root/class_name/video.mp4` (subdirectory name = class label).

```bash
python src/train.py -c src/experiments/base_config.yml \
    -d /path/to/ff++ \
    -d /path/to/celebdf
```

With a separate val/test dataset:

```bash
python src/train.py -c src/experiments/base_config.yml \
    -d /path/to/train_dataset \
    -vd /path/to/val_dataset
```

### Mode 2 — CSV-based multi-task datasets

CSV columns: `filename`, `target` (fake/real), `gender`, `ethnicity`, `emotion`.  
Use `meta_config.yml` to enable auxiliary heads.

```bash
python src/train.py -c src/experiments/meta_config.yml \
    -mc train_meta_v5.csv \
    --root_dir /path/to/videos
```

### Resume from checkpoint

```bash
python src/train.py -c src/experiments/base_config.yml \
    -d /path/to/dataset \
    -r checkpoints/last.ckpt
```

### FAU-only or rPPG-only training

```bash
python src/train_fau.py -c src/experiments/fau_config.yml -d /path/to/dataset
python src/train_rppg.py -c src/experiments/rppg_config.yml -d /path/to/dataset
```

### Key training parameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Main LR | 1e-4 |
| Encoder LR | 1e-5 (when `full_train=true`) |
| Weight decay | 0.05 |
| Scheduler | CosineAnnealingLR (T_max=100) |
| Early stopping | val_auc, patience=15 |
| Grad accumulation | 2 batches |
| Max epochs | 1000 |
| Checkpointing | best val_auc, last |

## Evaluation

```bash
python evaluate.py -c src/experiments/base_config.yml \
    -ckpt checkpoints/best.ckpt \
    -ed /path/to/test_dataset
```

Three evaluation modes:

| Flag | Mode |
|---|---|
| `-d /path` | Reproduce the training split (grouped by default, or `--legacy_split`), evaluate on `--split val\|test` |
| `-ed /path` | Evaluate the full dataset directly (no split) |
| `-mc meta.csv` | Evaluate from a CSV file |

Save results to JSON:

```bash
python evaluate.py ... -o results.json
```

Decision threshold: by default the predicted class is `argmax` (≈0.5). Pass `--threshold/-t` to instead predict `fake` when `P(fake) ≥ t` (AUROC is threshold-independent). Useful for calibration — e.g. the mixed-data fine-tune tends to over-flag `fake` at 0.5, and `-t 0.75…0.8` rebalances. **Calibrate the threshold on a separate val set, not on the test set.**

```bash
python evaluate.py -c src/experiments/base_config.yml -ckpt best.ckpt -mc val.csv -t 0.8 -o results.json
```

## Notes

- This is a **research codebase**, not a production package.
- Pretrained backbone weights are required for reproducible results.
- Architecture diagram source: `docs/architecture.drawio`.
- The default grouped split is deterministic (seed=42) and sorts file traversal, so it reproduces across machines. Use `--legacy_split` only to reproduce checkpoints trained before the 2026-06-12 split fix.

## Citation

If you use this repository, please cite the project page or contact the author directly.

---

## Русский

Исследовательский репозиторий для **мультимодальной детекции дипфейков**. Модель совместно использует:

- **FAU-ветка** — признаки единиц действия лица на уровне кадров (Swin Transformer Tiny + MEGraphAU GNN)
- **rPPG-ветка** — физиологические признаки на уровне видео (PhysNet, сигнал объёмного пульса крови)
- **Q-Former слияние** — Transformer Decoder с 32 обучаемыми запросами, выполняющий кросс-внимание к обеим модальностям
- **Attention Pooler + MLP** — бинарная классификация (REAL / FAKE)
- **Опциональные мультизадачные головы** — вспомогательные классификаторы пола, этничности и эмоций

## Архитектура

![Архитектура MDF](docs/architecture.png)

Подробное описание компонентов: [docs/architecture.md](docs/architecture.md).

Модель обрабатывает видеоклип `[B, 3, T, H, W]` через две параллельные ветки:

1. **FAU-ветка (уровень кадров).** Каждый кадр пропускается через MEGraphAU (Swin-T + GNN). Признаки проецируются в `embed_dim`, после чего применяется позиционное кодирование **по каждому AU** отдельно — каждый из 12 AU получает собственную временну́ю траекторию. Сегментное вложение `0` помечает FAU-токены.

2. **rPPG-ветка (уровень видео).** Весь клип подаётся в PhysNet, выдающий последовательность признаков `[B, T, D_phys]`. Признаки проецируются в `embed_dim`, добавляется синусоидальное позиционное кодирование, сегментное вложение `1` помечает rPPG-токены.

Обе последовательности токенов конкатенируются и подаются как **memory** в 6-слойный Transformer Decoder. 32 обучаемых запроса выполняют кросс-внимание к этой памяти. Декодированные запросы агрегируются Attention Pooler (softmax-взвешенная сумма), затем проходят через LayerNorm + Dropout в бинарный классификатор.

При включённых мультизадачных головах (`num_gender/ethnicity/emotion_classes > 0`) агрегированный признак дополнительно подаётся в отдельные линейные головы, веса задач балансируются через **uncertainty weighting** (Kendall et al. 2018).

## Датасеты

Обучение и оценка на трёх публичных датасетах:

| Датасет | Описание |
|---|---|
| **FF++** (FaceForensics++) | Face-swap видео с несколькими методами манипуляции |
| **CelebDF** (Celeb-DeepFake) | Высококачественные дипфейки знаменитостей |
| **VCDF-X** | AI-генерированный контент с лицами |

Препроцессинг кропает лица детектором MTCNN; train/val/test разбиение — случайное 70/15/15 (`seed=42`). См. раздел **Результаты** ниже про оговорку о сплите (и групповой сплит-фикс), а также чем текущая модель отличается от первых результатов.

## Результаты — как читать этот раздел

В репозитории приведены результаты **двух разных пайплайнов обучения**. Прочитайте это перед таблицами.

**1. Текущая модель — та, что отдаётся на тестирование.** Предобученный для детекции дипфейков rPPG-энкодер (DeepFakesON-Phys) + FAU-ветка ME-GraphAU, обучены совместно на **смеси** FF++ + CelebDF + VCDF-X. Это релевантная модель (раздел *Текущая модель* ниже).

- Её in-domain цифры на **test-сплите** (FF++ / CelebDF / VCDF-X) выглядят почти идеально (0.97–1.00 AUROC), но **завышены из-за особенности сплита**: каждый датасет хранит аугментированные варианты одного ролика отдельными файлами, и файловый сплит поместил часть этих копий и в train, и в test (измерено: 34% всех test-роликов, до ~53% на CelebDF и FF++). Это *верхняя оценка*, а не точность на новых данных.
- **Честная** цифра — на полностью **отложенной выборке** (53 видео, по хэшам содержимого проверено, что не пересекается с обучением, сделаны другими генераторами): **0.815 accuracy / 0.844 AUROC**. Это реалистичное ожидание на новых, невиданных видео.
- Проблема сплита уже исправлена для будущего обучения (групповой сплит); детали в разделе *Текущая модель*.

**2. Legacy-абляция — старый пайплайн, только качественно.** Обучение на одном датасете, кросс-датасетная оценка, на более раннем пайплайне обучения/сплита и на **валидационном** сплите. Оставлена только как **мотивация**: показывает, что модель, обученная на одном датасете, **не** генерализуется на другие — именно поэтому текущая модель обучается на смеси. **Не сравнивайте эти числа напрямую с текущей моделью** (другой пайплайн, другой сплит, val, а не test).

> **Главное для того, кто оценивает модель:** ожидайте качество около отложенной выборки (**~0.82 accuracy / ~0.84 AUROC**) на действительно невиданных данных. Почти единичные in-domain цифры завышены и не должны использоваться как ожидаемая точность.

## Legacy-абляция — кросс-оценка одно-датасетных моделей

> ⚠️ **Первые результаты, старый пайплайн — только качественная мотивация.** Это исходные кросс-датасетные цифры (добавлены в коммите `b04112f`, 2026-03-30). Они с раннего пайплайна, который существенно отличается от текущей модели — **без кропа лица детектором** (кадры как есть), исходные (не предобученные для детекции дипфейков) бэкбоны, меньше кадров на ролик, без контрастного лосса и memory bank — и метрики на **валидационном** сплите (`seed=42`). Не сравнивайте абсолютные значения с текущей моделью (см. *Текущая модель* → *Что изменилось* ниже). Важен *паттерн*, а не числа: модель, обученная на одном датасете, проседает на двух других (внедиагональные ячейки), что и мотивирует смешанное обучение.

Каждая модель обучена на одном датасете и оценена на всех трёх.

### Accuracy

| Обучение \ Тест | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.8316** | 0.7616 | 0.5479 |
| **CelebDF** | 0.6464 | **0.9342** | 0.4944 |
| **VCDF-X** | 0.4842 | 0.4875 | **0.9269** |
| **Смесь (все)** | 0.8057 | **0.9728** | **0.9131** |

### F1 (macro)

| Обучение \ Тест | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.8622** | 0.7320 | 0.5489 |
| **CelebDF** | 0.3914 | **0.9602** | 0.2585 |
| **VCDF-X** | 0.3140 | 0.1692 | **0.9055** |
| **Смесь (все)** | 0.8077 | **0.9809** | **0.9073** |

### AUROC

| Обучение \ Тест | FF++ | CelebDF | VCDF-X |
|---|:---:|:---:|:---:|
| **FF++** | **0.9758** | 0.8166 | 0.5768 |
| **CelebDF** | 0.7538 | **0.9999** | 0.3458 |
| **VCDF-X** | 0.4497 | 0.5445 | **0.9799** |
| **Смесь (все)** | 0.9351 | **0.9981** | **0.9752** |

### Наблюдения

- Каждая модель показывает высокие результаты на своём домене (0.83–0.93 accuracy).
- Кросс-доменная генерализация слабая для одиночных датасетов — артефакты VCDF-X и CelebDF существенно отличаются от FF++.
- **Обучение на смеси** (FF++ + CelebDF + VCDF-X) даёт наилучшую генерализацию: 0.81 / 0.97 / 0.91 по accuracy и 0.94 / 0.998 / 0.975 по AUROC.

## Текущая модель — обучение на смеси

**Это модель, которая отдаётся на тестирование.** Обучена совместно на смеси VCDF-X + CelebDF + FF++; rPPG-ветка использует предобученный для детекции дипфейков энкодер (DeepFakesON-Phys), FAU-ветка — ME-GraphAU (Swin-Tiny). Конфиг `src/experiments/base_config.yml` (`num_frames=64`, `num_queries=64`).

**Что изменилось с первых результатов (коммит `b04112f`).** Прирост относительно legacy-абляции дала комбинация:

- **Предобученный для детекции дипфейков rPPG-энкодер** — DeepFakesON-Phys вместо исходного PhysNet.
- **Кроп лица детектором MTCNN** в препроцессинге (раньше кадры подавались как есть).
- **Контрастное / метрическое обучение с memory bank** на слитом эмбеддинге (`metric_loss_type`, `memory_bank_size`).
- **Больше кадров на ролик** — `num_frames` 32 → 64.
- **Обучение на смеси датасетов** (FF++ + CelebDF + VCDF-X) и прочая настройка.

Оценка на test-сплите (`evaluate.py -d ... -s test -od {1..3}` точно воспроизводит сплит обучения: тот же порядок датасетов, seed=42, поэтому test не пересекается с train), плюс **отложенная выборка**, не участвовавшая в обучении (проверено: ноль совпадений sha256 между её 53 видео и всеми 32 150 обучающими видео; другие пулы исходников и генераторы):

| Датасет | Сплит | N | Accuracy | F1 (macro) | AUROC | Loss |
|---|---|---:|:---:|:---:|:---:|:---:|
| VCDF-X | test | 2107 | 0.9752 | 0.9664 | 0.9974 | 0.0758 |
| CelebDF | test | 1590 | 0.9976 | 0.9975 | 1.0000 | 0.0248 |
| FF++ | test | 1126 | 1.0000 | 1.0000 | 1.0000 | 0.0261 |
| **Отложенная выборка** | **отложенная** | 53 | **0.8150** | **0.7876** | **0.8437** | 0.5313 |

**Оговорка — пересечение контента train/test в test-сплитах.** Сплит 70/15/15 делается на уровне *файлов*, а в датасетах аугментированные варианты одного исходного ролика лежат отдельными файлами. Измерено на точно воспроизведённом сплите (seed=42): **34.1% test-видео — аугментированные копии train-видео** (VCDF-X 10.3%, CelebDF 52.5%, FF++ 52.8%). Поверх этого фейки одного исходника/актёра попадают и в train, и в test. Почти идеальные цифры на FF++/CelebDF из-за этого завышены; **честная оценка генерализации — отложенная выборка (0.815 acc / 0.844 AUROC)**.

Исправлено с 2026-06-12: `train.py`/`evaluate.py` по умолчанию используют **групповой сплит** — все варианты одного исходного ролика остаются по одну сторону сплита, обход файлов отсортирован (детерминирован между файловыми системами). Чекпоинты, обученные до исправления, оценивать с `--legacy_split`.

## Структура репозитория

```
src/
  models/
    rppg_p_fau.py           # DeepfakeDetector — основная модель
    rppg_p_fau_lightning.py # FauRPPGDeepFakeRecognizer — Lightning-модуль (мультизадачный)
    fau_classifier.py       # Классификатор только на FAU
    fau_lightning.py        # Lightning-модуль только FAU
    rppg_classifier.py      # Классификатор только на rPPG
    rppg_lightning.py       # Lightning-модуль только rPPG
  backbones/
    fau_encoder.py          # FAUEncoder — обёртка над MEGraphAU (MEFARG)
    rppg_encoder.py         # RPPGEncoder — обёртка над PhysNet
    pos.py                  # Синусоидальное позиционное кодирование
    MEGraphAU/              # Сабмодуль ME-GraphAU (Swin + GNN для AU)
    rPPGToolbox/            # Сабмодуль rPPG-Toolbox (PhysNet и другие)
  data/
    dataset.py              # VideoFolderDataset — загрузка из папки
    meta_dataset.py         # MetaVideoDataset — CSV-based мультизадачная загрузка
    transforms.py           # VideoTransform — согласованные аугментации
    processor.py            # FaceDetector (MTCNN) + Processor
    split.py                # экспериментальная утилита сплита (не используется)
  pooler/
    attn_pooler.py          # AttentionPooler — взвешенная агрегация
  loss/
    contrastive.py          # InfoNCEConsistencyLoss
  experiments/
    base_config.yml         # Стандартный конфиг (без вспомогательных голов)
    meta_config.yml         # Мультизадачный конфиг (gender=2, ethnicity=5, emotion=8)
    fau_config.yml          # Конфиг только для FAU
    rppg_config.yml         # Конфиг только для rPPG
  train.py                  # Основной скрипт обучения
  train_fau.py              # Обучение только FAU
  train_rppg.py             # Обучение только rPPG
  eval.py                   # Визуализация GradCAM / признаков
evaluate.py                 # Скрипт оценки (три режима датасетов)
env.sh                      # Интерактивная загрузка весов
load.py                     # Вспомогательные функции загрузки весов
docs/
  architecture.md           # Подробное описание архитектуры
  architecture.drawio       # Исходник диаграммы
  architecture.png          # Диаграмма архитектуры
  val_*.png                 # Графики метрик валидации
```

## Установка

### 1. Зависимости

```bash
uv sync
```

Или через pip:

```bash
pip install -e .
```

### 2. Загрузка предобученных весов

> Нужно **только для обучения с нуля**. Для инференса/оценки по обученному `.ckpt` этот шаг можно пропустить — чекпоинт уже содержит все веса (файлы swin/ME-GraphAU `.pth` загружаются только при наличии).

Интерактивный скрипт для скачивания весов FAU и backbone:

```bash
bash env.sh
```

Разместите веса FAU:
```
src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth
```

Разместите веса rPPG (из [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)):
```
src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth
```

## Обучение

Обучение реализовано на **PyTorch Lightning** с поддержкой DDP.

### Режим 1 — датасеты из папок

Структура: `root/class_name/video.mp4` (имя поддиректории = метка класса).

```bash
python src/train.py -c src/experiments/base_config.yml \
    -d /path/to/ff++ \
    -d /path/to/celebdf
```

С отдельным val/test датасетом:

```bash
python src/train.py -c src/experiments/base_config.yml \
    -d /path/to/train_dataset \
    -vd /path/to/val_dataset
```

### Режим 2 — CSV с мультизадачными метками

Колонки CSV: `filename`, `target` (fake/real), `gender`, `ethnicity`, `emotion`.  
Используйте `meta_config.yml` для включения вспомогательных голов.

```bash
python src/train.py -c src/experiments/meta_config.yml \
    -mc train_meta_v5.csv \
    --root_dir /path/to/videos
```

### Продолжение обучения из чекпоинта

```bash
python src/train.py -c src/experiments/base_config.yml \
    -d /path/to/dataset \
    -r checkpoints/last.ckpt
```

### Обучение только FAU или только rPPG

```bash
python src/train_fau.py -c src/experiments/fau_config.yml -d /path/to/dataset
python src/train_rppg.py -c src/experiments/rppg_config.yml -d /path/to/dataset
```

### Ключевые параметры

| Параметр | Значение |
|---|---|
| Оптимизатор | AdamW |
| LR основной | 1e-4 |
| LR энкодеров | 1e-5 (при `full_train=true`) |
| Weight decay | 0.05 |
| Планировщик | CosineAnnealingLR (T_max=100) |
| Early stopping | val_auc, patience=15 |
| Grad accumulation | 2 батча |
| Макс. эпох | 1000 |
| Чекпоинты | best val_auc + last |

## Оценка

```bash
python evaluate.py -c src/experiments/base_config.yml \
    -ckpt checkpoints/best.ckpt \
    -ed /path/to/test_dataset
```

Три режима оценки:

| Флаг | Режим |
|---|---|
| `-d /path` | Воспроизводит сплит обучения (по умолчанию групповой, или `--legacy_split`), оценивает `--split val\|test` |
| `-ed /path` | Оценивает весь датасет без разбиения |
| `-mc meta.csv` | Оценивает по CSV-файлу |

Сохранение результатов в JSON:

```bash
python evaluate.py ... -o results.json
```

Порог решения: по умолчанию класс берётся через `argmax` (≈0.5). Флаг `--threshold/-t` заставляет предсказывать `fake` при `P(fake) ≥ t` (AUROC от порога не зависит). Полезно для калибровки — дотюн на смеси склонен перекосить в `fake` при 0.5, а `-t 0.75…0.8` выравнивает. **Калибровать порог на отдельной val-выборке, не на тесте.**

```bash
python evaluate.py -c src/experiments/base_config.yml -ckpt best.ckpt -mc val.csv -t 0.8 -o results.json
```

## Примечания

- Это **исследовательский код**, не production-пакет.
- Предобученные веса backbone обязательны для воспроизводимых результатов.
- Исходник диаграммы архитектуры: `docs/architecture.drawio`.
- Групповой сплит по умолчанию детерминирован (seed=42) и сортирует обход файлов, поэтому воспроизводится между машинами. `--legacy_split` нужен только для воспроизведения чекпоинтов, обученных до исправления сплита 2026-06-12.

## Цитирование

Если вы используете этот репозиторий, ссылайтесь на страницу проекта или свяжитесь с автором.
