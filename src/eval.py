import cv2
import torch
import numpy as np
import typer
import functools # <--- БЫЛ ЗАБЫТ
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision import transforms
from omegaconf import OmegaConf

# Убедись, что путь к модели правильный
from src.models.rppg_p_fau_lightning import FauRPPGDeepFakeRecognizer

app = typer.Typer(pretty_exceptions_show_locals=False)

def videomae_reshape_transform(tensor, width=14, height=14):
    """Превращает токены VideoMAE [B, 1568, 768] обратно в картинку [B, 768, 14, 14]"""
    # Удаляем CLS токен если он есть
    result = tensor[:, 1:, :] if tensor.shape[1] == 1569 else tensor
        
    B, Seq, Dim = result.shape
    spatial = width * height
    temporal = Seq // spatial
    
    # [B, T, H*W, D] -> mean(T) -> [B, H*W, D]
    result = result.reshape(B, temporal, spatial, Dim).mean(dim=1)
    
    # [B, H, W, D] -> [B, D, H, W]
    result = result.reshape(B, height, width, Dim).permute(0, 3, 1, 2)
    return result

class Visualizer:
    def draw_graph(self, data, title, color, width=300, height=100, y_lim=None):
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        
        if len(data) > 1:
            d_arr = np.array(data)
            norm_data = (d_arr - d_arr.mean()) / (d_arr.std() + 1e-6)
        else:
            norm_data = data
            
        ax.plot(norm_data, color=color, linewidth=2)
        ax.set_title(title, fontsize=10, color='white', pad=-10)
        ax.set_facecolor('black')
        ax.axis('off')
        if y_lim: ax.set_ylim(y_lim)
        fig.patch.set_facecolor('black')
        plt.tight_layout(pad=0)
        
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img

    def draw_au_bars(self, au_logits, au_names, width=300, height=150):
        probs = torch.sigmoid(au_logits).cpu().detach().numpy().flatten()
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        bar_h = height // len(au_names)
        for i, (name, prob) in enumerate(zip(au_names, probs)):
            if i >= len(au_names): break
            color = (0, 255, 0) if prob > 0.5 else (100, 100, 100)
            length = int(prob * (width - 60))
            
            # Текст и полоска
            cv2.putText(img, name, (5, i*bar_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
            cv2.rectangle(img, (50, i*bar_h+2), (50+length, i*bar_h+bar_h-2), color, -1)
        return img

@app.command()
def process(
    video_path: str = typer.Option(..., "-i", help="Input video"),
    output_path: str = typer.Option("output_viz.mp4", "-o", help="Output video"),
    ckpt_path: str = typer.Option(..., "-c", help="Checkpoint path"),
    config_path: str = typer.Option("config.yaml", "-cfg", help="Config path"),
    device: str = typer.Option("cuda", help="cuda/mps/cpu")
):
    print(f"Loading from {ckpt_path}...")
    
    file_config = OmegaConf.load(config_path)
    defaults = {
        'backbone_fau': 'swin_transformer_tiny', 
        'num_frames': 128, # <-- Убедись что тут совпадает с буфером (128)
        'num_classes': 2,
        'dropout': 0.1, 
        'videomae_model_name': 'MCG-NJU/videomae-base',
        'num_au_classes': 12, 
        'lora_cfg': None
    }
    model_params = {**defaults, **OmegaConf.to_container(file_config.model_params, resolve=True)}
    
    lit_model = FauRPPGDeepFakeRecognizer.load_from_checkpoint(
        ckpt_path, model_params=model_params, map_location=device
    )
    lit_model.eval()
    lit_model.to(device)
    
    lit_model.forward = functools.partial(lit_model.forward, return_info=False)

    target_layer = lit_model.model.videomae.encoder.layer[-1]


    cam = GradCAM(
        model=lit_model, 
        target_layers=[target_layer], 
        reshape_transform=videomae_reshape_transform
    )

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    viz = Visualizer()
    rppg_buffer = []
    frames_buffer = []
    raw_buffer = []
    au_names = [f"AU{i}" for i in range(1, 13)]
    

    BATCH_SIZE = 128 
    
    print(f"Processing {total_frames} frames in batches of {BATCH_SIZE}...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(Image.fromarray(rgb))
        
        frames_buffer.append(tensor)
        raw_buffer.append(frame)
        
        if len(frames_buffer) == BATCH_SIZE:
            input_tensor = torch.stack(frames_buffer).permute(1,0,2,3).unsqueeze(0).to(device)
            
            with torch.no_grad():
                info_out = lit_model.model(input_tensor, return_info=True) 
                
                rppg_sig = info_out["rPPG"][0].cpu().numpy() 
                au_logits = info_out["au_logits"].mean(dim=0)
                logits = info_out["logits"]
                probs = torch.softmax(logits, dim=1)
                

            target_cls = 0 if probs[0,0] > 0.5 else 1 
            targets = [ClassifierOutputTarget(target_cls)]
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            heatmap = cv2.resize(grayscale_cam, (width, height))
            
            fake_score = probs[0,0].item()
            label_text = f"FAKE: {fake_score:.2f}" if fake_score > 0.5 else f"REAL: {1-fake_score:.2f}"
            color_text = (0,0,255) if fake_score > 0.5 else (0,255,0)
            
            for i in range(BATCH_SIZE):
                orig = raw_buffer[i]
                
                orig_float = orig.astype(np.float32) / 255.0
                vis = show_cam_on_image(cv2.cvtColor(orig_float, cv2.COLOR_BGR2RGB), heatmap, use_rgb=True)
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                
                rppg_buffer.append(rppg_sig[i])
                if len(rppg_buffer) > 100: rppg_buffer.pop(0)
                graph_img = viz.draw_graph(rppg_buffer, "rPPG Pulse", "#00ff00")
                
                au_img = viz.draw_au_bars(au_logits, au_names)
                
                h_g, w_g, _ = graph_img.shape
                h_a, w_a, _ = au_img.shape
                
                
                vis[height-h_g-10:height-10, width-w_g-10:width-10] = graph_img
                vis[60:60+h_a, 10:10+w_a] = au_img
                
                # Текст
                cv2.rectangle(vis, (5, 5), (300, 50), (0,0,0), -1)
                cv2.putText(vis, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_text, 3)
                
                out.write(vis)
                
            frames_buffer = []
            raw_buffer = []
            print(f"Batch processed. {label_text}")
            
    cap.release()
    out.release()
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    app()