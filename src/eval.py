import os
# Лечим фрагментацию памяти
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import torch
import torch.nn as nn
import numpy as np
import typer
import gc
from PIL import Image 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision import transforms
from omegaconf import OmegaConf

from src.models.rppg_p_fau_lightning import FauRPPGDeepFakeRecognizer

app = typer.Typer(pretty_exceptions_show_locals=False)

# --- 1. ОБЕРТКИ ---
class GradCAMModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x, return_info=False)

class VideoGradCAM(GradCAM):
    def get_target_width_height(self, input_tensor):
        if len(input_tensor.shape) == 5:
            return input_tensor.shape[-1], input_tensor.shape[-2]
        return super().get_target_width_height(input_tensor)

    def compute_cam_per_layer(self, input_tensor, targets, eigen_smooth):
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for i in range(len(activations_list)):
            target_layer = self.target_layers[i]
            layer_activations = activations_list[i]
            layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            cam = cam.astype(np.float32)

            if len(cam.shape) == 4: cam = cam[0]
            
            if len(cam.shape) == 3: # VIDEO
                scaled_video = []
                for t in range(cam.shape[0]):
                    frame = cv2.resize(cam[t], target_size)
                    scaled_video.append(frame)
                scaled = np.stack(scaled_video, axis=0)[None, ...] 
                cam_per_target_layer.append(scaled)
            else: # IMAGE
                scaled = cv2.resize(cam, target_size)
                cam_per_target_layer.append(scaled[None, None, ...])

        return cam_per_target_layer

# --- 2. РЕШЕЙПЕРЫ ---
def transformer_reshape_transform(tensor, width=14, height=14):
    if isinstance(tensor, tuple): tensor = tensor[0]
    if len(tensor.shape) == 4: return tensor 
    
    num_tokens = tensor.shape[1]
    spatial = width * height
    if num_tokens % spatial != 0: tensor = tensor[:, 1:, :]

    B, Seq, Dim = tensor.shape
    temporal = Seq // spatial
    valid_len = temporal * spatial
    if Seq > valid_len: tensor = tensor[:, :valid_len, :]
    
    result = tensor.reshape(B, temporal, spatial, Dim)
    side = int(np.sqrt(spatial))
    result = result.permute(0, 3, 1, 2).reshape(B, Dim, temporal, side, side)
    return result

def swin_reshape_transform(tensor, width=7, height=7):
    if isinstance(tensor, tuple): tensor = tensor[0]
    B_times_T, num_tokens, Dim = tensor.shape
    side = int(np.sqrt(num_tokens))
    result = tensor.permute(0, 2, 1).reshape(B_times_T, Dim, side, side)
    result = result.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return result

# --- 3. ВИЗУАЛИЗАТОР ---
class Visualizer:
    def draw_graph_strip(self, data, title, color, width, height):
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        
        if len(data) > 1:
            d_arr = np.array(data)
            std_val = d_arr.std() if d_arr.std() > 1e-6 else 1e-6
            norm_data = (d_arr - d_arr.mean()) / std_val
        else:
            norm_data = data
            
        ax.plot(norm_data, color=color, linewidth=2)
        ax.set_facecolor('black')
        ax.axis('off')
        ax.set_xlim(0, len(data) if len(data) > 0 else 1)
        ax.text(0.01, 0.85, title, transform=ax.transAxes, color='white', fontsize=12, weight='bold')

        fig.patch.set_facecolor('black')
        plt.tight_layout(pad=0)
        
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())
        img = img.astype(np.uint8)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close(fig)
        
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        return img

@app.command()
def process(
    video_path: str = typer.Option(..., "-i", help="Input video"),
    output_prefix: str = typer.Option("viz_result", "-o", help="Prefix"),
    ckpt_path: str = typer.Option(..., "-c", help="Checkpoint"),
    config_path: str = typer.Option("config.yaml", "-cfg", help="Config"),
    device: str = typer.Option("cuda", help="cuda/mps/cpu")
):
    torch.set_grad_enabled(True)
    
    print(f"Loading model from {ckpt_path}...")
    file_config = OmegaConf.load(config_path)
    defaults = {'backbone_fau': 'swin_transformer_tiny', 'num_frames': 128, 'num_classes': 2, 'dropout': 0.1, 'videomae_model_name': 'MCG-NJU/videomae-base', 'num_au_classes': 12, 'lora_cfg': None}
    model_params = {**defaults, **OmegaConf.to_container(file_config.model_params, resolve=True)}
    
    lit_model = FauRPPGDeepFakeRecognizer.load_from_checkpoint(ckpt_path, model_params=model_params, map_location=device)
    for param in lit_model.parameters(): param.requires_grad = True
    lit_model.eval()
    lit_model.to(device)
    
    cam_model = GradCAMModelWrapper(lit_model.model)
    target_mae = lit_model.model.videomae.encoder.layer[-1]
    try: target_fau = lit_model.model.au_encoder.backbone.layers[-1]
    except: target_fau = lit_model.model.au_proj

    cam_mae = VideoGradCAM(model=cam_model, target_layers=[target_mae], reshape_transform=transformer_reshape_transform)
    cam_fau = VideoGradCAM(model=cam_model, target_layers=[target_fau], reshape_transform=swin_reshape_transform)

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_mae, out_fau = None, None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    viz = Visualizer()
    rppg_buffer = []
    frames_buffer = []
    raw_buffer = []
    BATCH_SIZE = 128 
    GRAPH_HEIGHT = 150

    def process_and_write_batch(frames_tens, raw_frames, rppg_buf, w_mae, w_fau):
        real_len = len(frames_tens)
        padded_frames = frames_tens.copy()
        if real_len < 128:
            while len(padded_frames) < 128:
                needed = 128 - len(padded_frames)
                padded_frames.extend(frames_tens[:needed])
        
        input_tensor = torch.stack(padded_frames).permute(1,0,2,3).unsqueeze(0).to(device)
        input_tensor.requires_grad = True
            
        with torch.no_grad():
            info_out = lit_model.model(input_tensor, return_info=True) 
            rppg_sig = info_out["rPPG"][0].cpu().numpy() 
            probs = torch.softmax(info_out["logits"], dim=1)
            
            # === ФИНАЛЬНАЯ ЛОГИКА ===
            # prob[0] -> FAKE
            # prob[1] -> REAL
            print(probs)
            score_real = probs[0, 1].item()
            score_fake = probs[0, 0].item()
            
            if score_real > 0.5:
                pred_cls = 1 # Смотрим, почему это REAL
                label_text = f"REAL: {score_real:.2f}"
                color_text = (0, 255, 0) # Зеленый
            else:
                pred_cls = 0 # Смотрим, почему это FAKE
                label_text = f"FAKE: {score_fake:.2f}"
                color_text = (0, 0, 255) # Красный

        targets = [ClassifierOutputTarget(pred_cls)]
        
        with torch.amp.autocast('cuda'):
            gray_mae = cam_mae(input_tensor=input_tensor, targets=targets)
            if len(gray_mae.shape) == 4: gray_mae = gray_mae[0]
        torch.cuda.empty_cache(); gc.collect()

        with torch.amp.autocast('cuda'):
            gray_fau = cam_fau(input_tensor=input_tensor, targets=targets)
            if len(gray_fau.shape) == 4: gray_fau = gray_fau[0]
        torch.cuda.empty_cache(); gc.collect()

        frame_h, frame_w = raw_frames[0].shape[:2]
        total_height = frame_h + GRAPH_HEIGHT
        
        if w_mae is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            w_mae = cv2.VideoWriter(f"{output_prefix}_videomae.avi", fourcc, orig_fps, (frame_w, total_height))
            w_fau = cv2.VideoWriter(f"{output_prefix}_fau.avi", fourcc, orig_fps, (frame_w, total_height))

        for i in range(real_len):
            orig = raw_frames[i]
            if orig.shape[:2] != (frame_h, frame_w): orig = cv2.resize(orig, (frame_w, frame_h))
            
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            orig_rgb_float = orig_rgb.astype(np.float32) / 255.0
            
            def create_vis(cam_frames, idx):
                idx = idx if idx < len(cam_frames) else -1
                heatmap = cam_frames[idx]
                min_val, max_val = np.min(heatmap), np.max(heatmap)
                if max_val > min_val:
                    heatmap = (heatmap - min_val) / (max_val - min_val)
                
                heatmap = cv2.resize(heatmap, (frame_w, frame_h))
                v = show_cam_on_image(orig_rgb_float, heatmap, use_rgb=True)
                v = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
                return v

            vis_mae = create_vis(gray_mae, i)
            vis_fau = create_vis(gray_fau, i)
            
            rppg_val = rppg_sig[i] if i < len(rppg_sig) else 0
            rppg_buf.append(rppg_val)
            if len(rppg_buf) > 100: rppg_buf.pop(0)
            
            graph_strip = viz.draw_graph_strip(rppg_buf, "rPPG Pulse", "#00ff00", width=frame_w, height=GRAPH_HEIGHT)
            
            for v_frame, writer in [(vis_mae, w_mae), (vis_fau, w_fau)]:
                cv2.rectangle(v_frame, (5, 5), (300, 60), (0,0,0), -1)
                cv2.putText(v_frame, label_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_text, 3)
                combined = np.vstack((v_frame, graph_strip))
                writer.write(combined)
            
        print(f"Batch processed ({real_len} frames).")
        del input_tensor, gray_mae, gray_fau
        torch.cuda.empty_cache()
        return rppg_buf, w_mae, w_fau

    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(rgb))
        frames_buffer.append(tensor)
        raw_buffer.append(frame)
        
        if len(frames_buffer) == BATCH_SIZE:
            rppg_buffer, out_mae, out_fau = process_and_write_batch(frames_buffer, raw_buffer, rppg_buffer, out_mae, out_fau)
            frames_buffer, raw_buffer = [], []
            
    if len(frames_buffer) > 0:
        rppg_buffer, out_mae, out_fau = process_and_write_batch(frames_buffer, raw_buffer, rppg_buffer, out_mae, out_fau)

    cap.release()
    if out_mae: out_mae.release()
    if out_fau: out_fau.release()
    print(f"Done!")

if __name__ == "__main__":
    app()