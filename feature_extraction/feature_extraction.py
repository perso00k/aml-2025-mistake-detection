import argparse
import os
import glob
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Aggiunge la cartella corrente (feature_extraction) al path
# Così Python trova il 'core' di Perception Encoder che hai messo qui dentro
sys.path.append(os.getcwd())

try:
    from core.vision_encoder import pe
    from core.vision_encoder import transforms as pe_transforms
    from decord import VideoReader, cpu
except ImportError as e:
    print(f"ERRORE: {e}")
    print("Assicurati di lanciare lo script da DENTRO la cartella 'feature_extraction'.")
    sys.exit(1)

class PerceptionEncoderWrapper:
    def __init__(self, config_name, device):
        self.device = device
        print(f"Caricamento modello: {config_name}...")
        self.model = pe.CLIP.from_config(config_name, pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.preprocess = pe_transforms.get_image_transform(self.model.image_size)

    def extract(self, frames_pil):
        images = torch.stack([self.preprocess(f) for f in frames_pil]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(images, normalize=True)
        return features.cpu()

def process_video(video_path, model_wrapper, args):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        # Campiona frame in base agli FPS richiesti
        step = max(1, vr.get_avg_fps() / args.fps)
        indices = np.arange(0, total_frames, step).astype(int)
        
        all_features = []
        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i : i + args.batch_size]
            frames_np = vr.get_batch(batch_idx).asnumpy()
            frames_pil = [Image.fromarray(f) for f in frames_np]
            
            feats = model_wrapper.extract(frames_pil)
            all_features.append(feats)
            
        return torch.cat(all_features, dim=0) if all_features else None
    except Exception as e:
        print(f"Errore su {video_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="../videos", help="Dove sono i video mp4")
    parser.add_argument("--output_dir", type=str, default="../features/PE_features", help="Dove salvare i file .npz")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--model", type=str, default="PE-Core-B16-224")
    args = parser.parse_args()

    # Controllo di sicurezza per i percorsi relativi
    if args.video_dir.startswith(".."):
        base_dir = os.path.dirname(os.getcwd())
        args.video_dir = os.path.join(base_dir, args.video_dir.strip("../"))
        args.output_dir = os.path.join(base_dir, args.output_dir.strip("../"))

    print(f"Leggo video da: {args.video_dir}")
    print(f"Salvo feature in: {args.output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PerceptionEncoderWrapper(args.model, device)
    
    videos = glob.glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True)
    print(f"Trovati {len(videos)} video.")

    for vid in tqdm(videos):
        rel_path = os.path.relpath(vid, args.video_dir)
        
        save_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + ".npz")
        
        if os.path.exists(save_path): continue
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        feats = process_video(vid, model, args)
        
        if feats is not None:
            if isinstance(feats, torch.Tensor):
                feats_np = feats.detach().cpu().numpy()
            else:
                feats_np = np.array(feats)

            np.savez_compressed(save_path, features=feats_np)

if __name__ == "__main__":
    main()