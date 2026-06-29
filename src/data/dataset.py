import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# OpenCV's internal thread pool busy-spins after fork() in DataLoader workers
# (workers stuck ~90% CPU, no batches, GPU idle, training hangs on corrupt H.264).
# Disable it process-wide; forked workers inherit this. Standard fix for cv2+DataLoader.
cv2.setNumThreads(0)

import signal
import threading


class _VideoReadTimeout(Exception):
    """Raised by SIGALRM when a video read exceeds its time budget (corrupt H.264)."""


def _video_alarm_handler(signum, frame):
    raise _VideoReadTimeout()

class RecursiveFolderDataset(Dataset):
    """
    for ff++ and celebDF
    """
    def __init__(self, root_dir, transform=None, valid_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Folder {root_dir} not found")

        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for idx, class_name in enumerate(subdirs):
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx

            class_folder = os.path.join(root_dir, class_name)

            for root, _, files in os.walk(class_folder):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        path = os.path.join(root, file)
                        self.samples.append((path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Ошибка при загрузке {path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label



def grouped_split_indices(paths, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Сплит 70/15/15, не разрывающий группы вариантов одного ролика.

    Файлы вида ``X.mp4`` / ``X_blur_compression.mp4`` / ``X_brightness_gamma.mp4``
    считаются одним исходным роликом и целиком попадают в одну часть сплита.
    Группа определяется по стему имени: если стем файла продолжает через '_'
    стем другого существующего файла, оба сводятся к общему базовому стему.

    Returns (train_idx, val_idx, test_idx).
    """
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    stem_set = set(stems)

    def base_of(stem):
        reduced = True
        while reduced:
            reduced = False
            for i in range(len(stem) - 1, 0, -1):
                if stem[i] == "_" and stem[:i] in stem_set:
                    stem = stem[:i]
                    reduced = True
                    break
        return stem

    groups = {}
    for i, s in enumerate(stems):
        groups.setdefault(base_of(s), []).append(i)

    keys = sorted(groups)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(keys), generator=g).tolist()

    n_tr = int(train_ratio * len(paths))
    n_va = int(val_ratio * len(paths))
    train_idx, val_idx, test_idx = [], [], []
    for k in order:
        bucket = groups[keys[k]]
        if len(train_idx) < n_tr:
            train_idx.extend(bucket)
        elif len(val_idx) < n_va:
            val_idx.extend(bucket)
        else:
            test_idx.extend(bucket)
    return train_idx, val_idx, test_idx


class VideoFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, video_transform=None,
                 valid_extensions=('.mp4', '.avi', '.mov', '.mkv'), frames_per_video=32,
                 sort_files=True):
        self.root_dir = root_dir
        self.transform = transform
        self.video_transform = video_transform
        self.valid_extensions = valid_extensions
        self.frames_per_video = frames_per_video
        # sort_files=False — порядок os.walk как до 2026-06-12, нужен только
        # для воспроизведения сплита старых чекпоинтов (--legacy_split).
        self.sort_files = sort_files

        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Folder {root_dir} not found")
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) if d !='captions'])
        for idx, class_name in enumerate(subdirs):
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx
            class_folder = os.path.join(root_dir, class_name)

            for root, dirs, files in os.walk(class_folder):
                if self.sort_files:
                    dirs.sort()
                    files = sorted(files)
                for file in files:
                    if file.lower().endswith(self.valid_extensions):
                        path = os.path.join(root, file)
                        self.samples.append((path, idx))

        print(f"Found {len(self.samples)} videos in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def _get_dummy_video(self):
        return [Image.new('RGB', (224, 224)) for _ in range(self.frames_per_video)]

    def _load_video(self, path):
        # SIGALRM timeout bounds hangs on corrupt H.264 (cv2.read can spin forever
        # in C; the signal interrupts even a blocking C call). Only arms in the
        # worker's main thread (where __getitem__ runs); otherwise reads as before.
        use_alarm = threading.current_thread() is threading.main_thread()
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _video_alarm_handler)
            signal.alarm(45)
        cap = None
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Error opening {path}")
                return self._get_dummy_video()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return self._get_dummy_video()

            clip_len = self.frames_per_video
            if total_frames > clip_len:
                start_frame = np.random.randint(0, total_frames - clip_len)
            else:
                start_frame = 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = []
            for _ in range(clip_len):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
                else:
                    break

            if len(frames) == 0:
                return self._get_dummy_video()

            original_len = len(frames)
            while len(frames) < clip_len:
                frames.append(frames[len(frames) % original_len])
            return frames[:clip_len]
        except _VideoReadTimeout:
            print(f"⏱️  TIMEOUT — пропускаю битое видео: {path}")
            return self._get_dummy_video()
        except Exception as e:
            print(f"Ошибка чтения {path}: {e}")
            return self._get_dummy_video()
        finally:
            if cap is not None:
                cap.release()
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            frames = self._load_video(path)
        except Exception as e:
            frames = self._get_dummy_video()

        if self.video_transform:
            video_tensor = self.video_transform(frames)
        elif self.transform:
            frames = [self.transform(img) for img in frames]
            video_tensor = torch.stack(frames).permute(1, 0, 2, 3)
        else:
            raise ValueError("No transform or video_transform provided")

        return video_tensor, label


class FrameFolderDataset(Dataset):
    """Pre-cropped face frames stored per clip (e.g. GenD preproc subsample).

    Layout:  root/<class_name>/<clip_id>/frame_XXXX.png
    Classes come from sorted subdir names → fake=0, real=1 (matches
    VideoFolderDataset and MetaVideoDataset.DEFAULT_TARGET_MAP).

    Frames are ALREADY face-cropped, so NO FaceDetector is applied — the
    video_transform is run directly on the loaded PIL frames. Returns the same
    (video_tensor [C,T,H,W], label) as VideoFolderDataset, so it can be mixed
    with it via ConcatDataset.

    Note: these frames are subsampled (irregular stride) → the rPPG temporal
    signal is unreliable; intended for use with the rPPG encoder frozen.
    """

    IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(self, root_dir, video_transform=None, frames_per_video=64):
        if video_transform is None:
            raise ValueError("FrameFolderDataset requires a video_transform")
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Folder {root_dir} not found")
        self.root_dir = root_dir
        self.video_transform = video_transform
        self.frames_per_video = frames_per_video

        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        subdirs = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
        for idx, class_name in enumerate(subdirs):
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name)
            for clip in sorted(os.listdir(class_dir)):
                clip_dir = os.path.join(class_dir, clip)
                if os.path.isdir(clip_dir):
                    self.samples.append((clip_dir, idx))
        print(f"FrameFolderDataset: {len(self.samples)} clips in {root_dir} (classes {subdirs})")

    def __len__(self):
        return len(self.samples)

    def _dummy(self):
        return [Image.new("RGB", (224, 224)) for _ in range(self.frames_per_video)]

    def _load_frames(self, clip_dir):
        files = sorted(f for f in os.listdir(clip_dir) if f.lower().endswith(self.IMG_EXT))
        if not files:
            return self._dummy()
        n = len(files)
        clip_len = self.frames_per_video
        # Evenly sample clip_len frames, preserving temporal order; pad by cycling.
        if n >= clip_len:
            idxs = np.linspace(0, n - 1, clip_len).astype(int)
            sel = [files[i] for i in idxs]
        else:
            sel = files + [files[i % n] for i in range(clip_len - n)]
        frames = []
        for f in sel:
            try:
                frames.append(Image.open(os.path.join(clip_dir, f)).convert("RGB"))
            except Exception as e:
                print(f"Ошибка чтения кадра {os.path.join(clip_dir, f)}: {e}")
                frames.append(Image.new("RGB", (224, 224)))
        return frames

    def __getitem__(self, idx):
        clip_dir, label = self.samples[idx]
        frames = self._load_frames(clip_dir)
        video_tensor = self.video_transform(frames)
        return video_tensor, label


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Need sum == 1"

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_set, val_set, test_set

if __name__ == "__main__":

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    #full_dataset = RecursiveFolderDataset(root_path, transform=data_transforms)

    # print(f"Classes: {full_dataset.classes}")
    # print(f"Len images: {len(full_dataset)}")
    # train_ds, val_ds, test_ds = split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    # train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # images, labels = next(iter(train_loader))
    # print(f": {images.shape}")
    # print(f"Labels: {labels}")


    #root_path = "/mnt/tank/scratch/dstoronkin/faigc_dataset/dataset/videos"
    
    root_path = "/mnt/tank/scratch/dstoronkin/ff++_videos_out"

    import json
    jsonn = '/mnt/tank/scratch/dstoronkin/ff++_videos_out/captions/results.json'
    
    #print(json.load(open(jsonn)))
    full_dataset = VideoFolderDataset(root_path, transform=data_transforms)

    print(f"Classes: {full_dataset.classes}")
    print(f"Len images: {len(full_dataset)}")
    train_ds, val_ds, test_ds = split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    images, labels = next(iter(train_loader))
    print(f": {images.shape}")
    print(f"Labels: {labels}")