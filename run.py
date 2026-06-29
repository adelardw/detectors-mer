"""
Инференс детектора дипфейков на папке видео → JSONL с предсказаниями.

Пример:
    uv run run.py \
        -ckpt path/to/checkpoint.ckpt \
        -c src/experiments/base_config.yml \
        -d /path/to/videos \
        -o predictions.jsonl

Каждая строка JSONL — dict по одному видео:
    {
      "video": "clip_001.mp4",          # путь относительно папки -d
      "label": "fake",                  # предсказанная метка класса
      "label_id": 0,                    # её индекс (см. LABEL_MAP)
      "prob": 0.97,                     # вероятность предсказанного класса
      "probs": {"fake": 0.97, "real": 0.03}
    }

Маппинг меток повторяет обучение: классы брались из имён папок real/fake,
отсортированных по алфавиту (VideoFolderDataset) → fake=0, real=1.
То же самое в MetaVideoDataset.DEFAULT_TARGET_MAP = {"fake": 0, "real": 1}.
"""
import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import typer
from omegaconf import OmegaConf
from typing import Optional
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.models.rppg_p_fau_df_lightning import FauRPPGDeepFakeRecognizerDF
from src.data.transforms import VideoTransform
from src.data.processor import FaceDetector, Processor

load_dotenv()
app = typer.Typer(pretty_exceptions_show_locals=False)

# Метки берутся из имён папок (sorted): fake=0, real=1 — как в VideoFolderDataset
# и MetaVideoDataset.DEFAULT_TARGET_MAP. Изменять только вместе с обучением.
LABEL_MAP = {0: "fake", 1: "real"}

VALID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Совпадает с DEFAULT_MODEL_CFG в evaluate.py — переопределяется значениями из конфига.
DEFAULT_MODEL_CFG = {
    "backbone_fau": "swin_transformer_tiny",
    "num_frames": 128,
    "au_ckpt_path": "./src/backbones/MEGraphAU/checkpoints/MEFARG_swin_tiny_BP4D_fold1.pth",
    "phys_ckpt_path": "./src/backbones/rPPGToolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth",
    "num_classes": 2,
    "dropout": 0.3,
    "num_au_classes": 12,
    "embed_dim": 512,
    "num_queries": 32,
    "num_decoder_layers": 6,
    "nhead": 8,
    "num_gender_classes": 0,
    "num_ethnicity_classes": 0,
    "num_emotion_classes": 0,
}


class InferenceVideoDataset(Dataset):
    """Плоский набор видео из папки (рекурсивно, отсортированно), без меток.

    Кадры сэмплируются ДЕТЕРМИНИРОВАННО (с центра ролика), чтобы один и тот же
    файл всегда давал одно и то же предсказание — это инструмент для сдачи.
    """

    def __init__(self, source, video_transform, frames_per_video=64):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Путь не найден: {source}")
        self.video_transform = video_transform
        self.frames_per_video = frames_per_video

        self.paths = []
        if os.path.isfile(source):
            # Одно видео: пути в JSONL будут просто именем файла.
            if not source.lower().endswith(VALID_EXTENSIONS):
                raise Exception(f"Не видеофайл ({VALID_EXTENSIONS}): {source}")
            self.root_dir = os.path.dirname(source) or "."
            self.paths.append(source)
        else:
            self.root_dir = source
            for root, dirs, files in os.walk(source):
                dirs.sort()
                for file in sorted(files):
                    if file.lower().endswith(VALID_EXTENSIONS):
                        self.paths.append(os.path.join(root, file))
        if not self.paths:
            raise Exception(f"Нет видео ({VALID_EXTENSIONS}) по пути: {source}")
        print(f"Найдено видео: {len(self.paths)} в {source}")

    def __len__(self):
        return len(self.paths)

    def _dummy(self):
        return [Image.new("RGB", (224, 224)) for _ in range(self.frames_per_video)]

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Ошибка открытия {path}")
            return self._dummy()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return self._dummy()

        clip_len = self.frames_per_video
        # Детерминированный старт: центр ролика (в отличие от случайного в обучении).
        start_frame = max(0, (total_frames - clip_len) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        for _ in range(clip_len):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        if len(frames) == 0:
            return self._dummy()
        original_len = len(frames)
        while len(frames) < clip_len:
            frames.append(frames[len(frames) % original_len])
        return frames[:clip_len]

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            frames = self._load_video(path)
        except Exception as e:
            print(f"Ошибка чтения {path}: {e}")
            frames = self._dummy()
        video_tensor = self.video_transform(frames)
        rel = os.path.relpath(path, self.root_dir)
        return video_tensor, rel


def _collate(batch):
    tensors = torch.stack([b[0] for b in batch])
    rels = [b[1] for b in batch]
    return tensors, rels


@app.command()
def run(
    ckpt_path: str = typer.Option(..., "--ckpt_path", "-ckpt", help="Путь к .ckpt чекпоинту"),
    config_path: str = typer.Option("src/experiments/base_config.yml", "--config_name", "-c", help="Путь к .yaml конфигу модели"),
    video_dir: str = typer.Option(..., "--video_dir", "-d", help="Папка с видео (рекурсивно) ИЛИ путь к одному видеофайлу"),
    output_path: str = typer.Option("predictions.jsonl", "--output", "-o", help="Куда писать JSONL"),
    batch_size: int = typer.Option(8, "--batch_size", "-bs"),
    num_workers: int = typer.Option(4, "--num_workers", "-nw"),
    no_face_detector: bool = typer.Option(False, "--no_face_detector", help="Отключить MTCNN-детектор лиц в препроцессинге"),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", "-t",
        help="Порог по P(fake): метка fake если P(fake)>=threshold, иначе real. По умолчанию (None) — argmax (≈0.5)."
    ),
):
    """Прогнать чекпоинт на папке видео и сохранить вероятности классов в JSONL."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")

    typer.echo(f"Config:     {config_path}")
    typer.echo(f"Checkpoint: {ckpt_path}")

    file_config = OmegaConf.load(config_path)
    final_config = OmegaConf.merge(OmegaConf.create(DEFAULT_MODEL_CFG), file_config.model_params)
    model_cfg = OmegaConf.to_container(final_config, resolve=True)

    num_frames = model_cfg["num_frames"]
    num_classes = model_cfg.get("num_classes", 2)
    typer.echo(f"num_frames={num_frames}, num_classes={num_classes}")
    if num_classes != len(LABEL_MAP):
        typer.echo(f"⚠️  num_classes={num_classes}, но LABEL_MAP описывает {len(LABEL_MAP)} меток.")

    # Идентичный train.py/evaluate.py препроцессинг (val-режим).
    val_transform = Processor(
        transform=VideoTransform(size=(224, 224), training=False),
        detector=None if no_face_detector else FaceDetector(margin=20, device="cpu"),
    )
    if no_face_detector:
        typer.echo("⚠️  Face detector ОТКЛЮЧЁН — кадры подаются как есть (resize 224×224).")

    ds = InferenceVideoDataset(video_dir, video_transform=val_transform, frames_per_video=num_frames)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0, collate_fn=_collate,
    )

    typer.echo("Загрузка чекпоинта...")
    model = FauRPPGDeepFakeRecognizerDF.load_from_checkpoint(
        ckpt_path, model_params=model_cfg, map_location="cpu", strict=False,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"Device: {device}")
    model = model.to(device)

    n_written = 0
    with open(output_path, "w", encoding="utf-8") as out, torch.no_grad():
        for batch_idx, (x, rels) in enumerate(loader):
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu()
            # argmax по умолчанию; при --threshold: fake(0) если P(fake)>=thr, иначе real(1)
            if threshold is None:
                preds = probs.argmax(dim=1)
            else:
                preds = torch.where(probs[:, 0] >= threshold, 0, 1)

            for i, rel in enumerate(rels):
                pid = int(preds[i])
                p = probs[i]
                record = {
                    "video": rel,
                    "label": LABEL_MAP.get(pid, str(pid)),
                    "label_id": pid,
                    "prob": round(float(p[pid]), 6),
                    "probs": {LABEL_MAP.get(c, str(c)): round(float(p[c]), 6) for c in range(num_classes)},
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                typer.echo(f"  [{batch_idx + 1}/{len(loader)}]  обработано {n_written}")

    typer.echo(f"\nГотово. Предсказаний записано: {n_written} → {output_path}")


if __name__ == "__main__":
    app()
