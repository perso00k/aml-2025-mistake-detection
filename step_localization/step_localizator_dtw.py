import os
import json
import numpy as np
import glob
from tqdm import tqdm

# --- CONFIGURAZIONE ---
VIDEO_FEATS_DIR = "/content/code/data/video/perception_encoder"
GRAPH_FEATS_DIR = "/content/code/data/graphs"
OUTPUT_DIR = "/content/code/data/localized_features"
RECIPES_MAP_FILE = "/content/code/recipe_map.json"  # Usiamo il file appena creato

def load_recipe_map():
    if not os.path.exists(RECIPES_MAP_FILE):
        print(f"❌ ERRORE: Manca il file {RECIPES_MAP_FILE}")
        return {}
    with open(RECIPES_MAP_FILE, 'r') as f:
        return json.load(f)

def load_data(video_path, recipe_map):
    # 1. Estrai ID video
    filename = os.path.basename(video_path)
    # Gestione nomi file: "1_10_360p.mp4_1s_1s.npz"-> "1_10"
    base_name = filename.split('.')[0].replace('_360p', '')
    parts = base_name.split('_')
    
    if len(parts) < 2: 
        return None, None, None, None # Nome file non standard
        
    video_id = f"{parts[0]}_{parts[1]}" # Es. "1_10"
    recipe_idx = parts[0]               # Es. "1"
    
    # 2. Trova nome ricetta
    recipe_name = recipe_map.get(recipe_idx)
    
    if not recipe_name:
        # print(f"⚠️ Nessuna ricetta mappata per ID {recipe_idx} (Video: {filename})")
        return None, None, None, None

    # 3. Carica Video
    try:
        d = np.load(video_path)
        v_feat = d['arr_0'] if 'arr_0' in d else d[d.files[0]]
    except Exception as e:
        print(f"Errore lettura video {video_path}: {e}")
        return None, None, None, None
    
    # 4. Carica Grafo
    # Cerca il file .npz usando il nome mappato
    g_path = os.path.join(GRAPH_FEATS_DIR, f"{recipe_name}.npz")
    
    if not os.path.exists(g_path):
        # Tentativo di debug: stampa i file disponibili se fallisce
        # print(f"⚠️ Grafo non trovato: {g_path} (Cercavo: {recipe_name}.npz)")
        return None, None, None, None
    
    g_data = np.load(g_path)
    step_ids = sorted([k for k in g_data.files if k.isdigit()], key=lambda x: int(x))
    if not step_ids: return None, None, None, None
    
    g_feats = [g_data[k].flatten() for k in step_ids]
    
    return video_id, v_feat, np.stack(g_feats), step_ids

def align_dtw(sim_matrix):
    """Calcola allineamento ottimo (DTW)."""
    n_frames, n_steps = sim_matrix.shape
    dp = np.full((n_frames, n_steps), -np.inf)
    backpointers = np.zeros((n_frames, n_steps), dtype=int)
    
    dp[0, 0] = sim_matrix[0, 0]
    
    for t in range(1, n_frames):
        for s in range(n_steps):
            score_same = dp[t-1, s]
            score_prev = dp[t-1, s-1] if s > 0 else -np.inf
            
            if score_same >= score_prev:
                dp[t, s] = score_same + sim_matrix[t, s]
                backpointers[t, s] = 0 
            else:
                dp[t, s] = score_prev + sim_matrix[t, s]
                backpointers[t, s] = 1 
                
    segments = []
    curr_s = n_steps - 1
    seg_end = n_frames
    
    for t in range(n_frames - 1, 0, -1):
        if backpointers[t, curr_s] == 1: 
            segments.append({'step_idx': curr_s, 'start': t, 'end': seg_end})
            seg_end = t
            curr_s -= 1
            if curr_s < 0: break
            
    if curr_s >= 0:
        segments.append({'step_idx': curr_s, 'start': 0, 'end': seg_end})
        
    segments.reverse()
    return segments

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    recipe_map = load_recipe_map()
    
    print(f"Mappa caricata: {len(recipe_map)} ricette.")
    
    # Cerca tutti i file video
    video_files = glob.glob(os.path.join(VIDEO_FEATS_DIR, "*.npz"))
    print(f"Trovati {len(video_files)} file video potenziali.")
    
    count_saved = 0
    count_missing_graph = 0
    
    for v_path in tqdm(video_files):
        video_id, v_feat, g_feat, step_ids_map = load_data(v_path, recipe_map)
        
        if video_id is None: 
            # Se load_data ritorna None, o manca la mappa o manca il grafo
            continue
        
        # --- LOGICA DTW ---
        v_norm = v_feat / (np.linalg.norm(v_feat, axis=1, keepdims=True) + 1e-8)
        g_norm = g_feat / (np.linalg.norm(g_feat, axis=1, keepdims=True) + 1e-8)
        sim_matrix = v_norm @ g_norm.T 
        
        segments = align_dtw(sim_matrix)
        
        step_embeddings = []
        detected_step_ids = []
        segments_data = [] 
        
        segments.sort(key=lambda x: x['step_idx'])
        
        for seg in segments:
            t_start, t_end = seg['start'], seg['end']
            real_id = step_ids_map[seg['step_idx']]
            
            segments_data.append([float(t_start), float(t_end), int(real_id)])

            if t_end > t_start:
                avg_feat = np.mean(v_feat[t_start:t_end], axis=0)
                step_embeddings.append(avg_feat)
                detected_step_ids.append(real_id)
        
        if not step_embeddings: continue
            
        # Salva
        out_path = os.path.join(OUTPUT_DIR, f"{video_id}.npz")
        np.savez_compressed(
            out_path, 
            features=np.stack(step_embeddings), 
            step_ids=np.array(detected_step_ids),
            segments=np.array(segments_data)
        )
        count_saved += 1

    print(f"\n✅ Finito!")
    print(f"   Video elaborati correttamente: {count_saved}")
    if (len(video_files) - count_saved) > 0:
        print(f"   Video saltati (ricetta non in mappa o grafo non trovato): {len(video_files) - count_saved}")
        print("   (Controlla che i nomi nel recipe_map.json coincidano coi nomi file in data/graphs)")

if __name__ == "__main__":
    main()