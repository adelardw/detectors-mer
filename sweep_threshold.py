import sys, numpy as np, torch, torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.models.rppg_p_fau_df_lightning import FauRPPGDeepFakeRecognizerDF
from src.data.meta_dataset import MetaVideoDataset
from src.data.transforms import VideoTransform
from src.data.processor import FaceDetector, Processor

CKPT = sys.argv[1] if len(sys.argv) > 1 else "finetune_e0.ckpt"
torch.manual_seed(42); np.random.seed(42)

cfg = OmegaConf.load("src/experiments/base_config.yml")
model_cfg = OmegaConf.to_container(cfg.model_params, resolve=True)
nf = model_cfg["num_frames"]

vt = Processor(transform=VideoTransform(size=(224, 224), training=False),
               detector=FaceDetector(margin=20, device="cpu"))
ds = MetaVideoDataset("hwei_part1/labels_eval.csv", video_transform=vt,
                      frames_per_video=nf, root_dir="hwei_part1")  # fake=0, real=1

model = FauRPPGDeepFakeRecognizerDF.load_from_checkpoint(
    CKPT, model_params=model_cfg, map_location="cpu", strict=False)
model.eval()

P, Y = [], []
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
with torch.no_grad():
    for x, t in loader:
        y = t["label"] if isinstance(t, dict) else t
        p = F.softmax(model(x), dim=1)
        P.append(p); Y.append(y)
probs = torch.cat(P).numpy()
y = torch.cat(Y).numpy().astype(int)          # 0=fake, 1=real
pfake = probs[:, 0]                            # P(fake)
print(f"N={len(y)} | fake(0)={int((y==0).sum())} real(1)={int((y==1).sum())}")


def metrics(pred):
    acc = (pred == y).mean()
    # macro-F1 over classes {0,1}
    f1s = []
    for c in (0, 1):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    fake_rec = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    real_rec = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    return acc, np.mean(f1s), fake_rec, real_rec


import json


def row(thr):
    pred = np.where(pfake >= thr, 0, 1)         # predict fake if P(fake)>=thr
    acc, f1m, fr, rr = metrics(pred)
    return {"threshold": round(float(thr), 2), "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1m), 4), "fake_recall": round(float(fr), 4),
            "real_recall": round(float(rr), 4)}


sweep = [row(thr) for thr in np.round(np.arange(0.05, 0.96, 0.05), 2)]
best = max(sweep, key=lambda r: r["accuracy"])
argmax05 = row(0.5)

print(f"{'thr(Pfake>=)':>12} {'acc':>6} {'f1m':>6} {'fakeRec':>8} {'realRec':>8}")
for r in sweep:
    star = "  <-- best acc" if r is best else ""
    print(f"{r['threshold']:>12.2f} {r['accuracy']:>6.4f} {r['f1_macro']:>6.4f} {r['fake_recall']:>8.4f} {r['real_recall']:>8.4f}{star}")

out = {
    "checkpoint": CKPT,
    "dataset": "hwei_part1 (held-out)",
    "num_samples": int(len(y)),
    "n_fake": int((y == 0).sum()),
    "n_real": int((y == 1).sum()),
    "label_map": {"fake": 0, "real": 1},
    "criterion": "predict fake if P(fake) >= threshold",
    "argmax_0.5": argmax05,
    "best_by_accuracy": best,
    "sweep": sweep,
    "caveat": "threshold tuned on the same 53-clip held-out set -> optimistic/overfit; calibrate on a separate val set for deployment",
}
with open("threshold_sweep_e0.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\n=== JSON ===")
print(json.dumps(out, indent=2, ensure_ascii=False))
print("\nсохранено: threshold_sweep_e0.json")
