import subprocess
import re
import ast
import sys
import os

# --- CONFIGURAZIONE ---
# Inserisci qui il percorso ESATTO del tuo checkpoint migliore
# CKPT_PATH = r"checkpoints_test\bigru_bs8_reg\error_recognition\BiGRU\omnivore\error_recognition_step_omnivore_BiGRU_video_best.pt"
# CKPT_PATH = r"checkpoints_test\bigru_bs8_reg\error_recognition\BiGRU\omnivore\error_recognition_step_omnivore_BiGRU_video_epoch_11.pt"
CKPT_PATH = r"checkpoints_test\bigru_bs8_reg_slowfast\error_recognition\BiGRU\slowfast\error_recognition_step_slowfast_BiGRU_video_epoch_19.pt"
VARIANT = "BiGRU"
BACKBONE = "slowfast"
SPLIT = "recordings"
# ----------------------

# Usa os.path.basename per ottenere il nome del file in modo sicuro
ckpt_filename = os.path.basename(CKPT_PATH)

print("-" * 75)
print(f" OTTIMIZZAZIONE SOGLIA per: {ckpt_filename}") 
print("-" * 75)
print(f"{'Thr':<5} | {'F1':<8} | {'Precision':<10} | {'Recall':<10} | {'AUC':<8} | {'Accuracy':<10}")
print("-" * 75)

best_f1 = 0
best_thr = 0

# Testiamo le soglie da 0.3 a 0.7
thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

for thresh in thresholds:
    # Costruiamo il comando
    cmd = [
        sys.executable, "-m", "core.evaluate",
        "--variant", VARIANT,
        "--backbone", BACKBONE,
        "--split", SPLIT,
        "--threshold", str(thresh),
        "--ckpt", CKPT_PATH
    ]
    
    try:
        # Esegue il comando e cattura l'output
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        output = result.stdout
        
        # Cerca la riga con le metriche
        match = re.search(r"test Step Level Metrics: ({.*?})", output)
        
        if match:
            dict_str = match.group(1)
            
            # --- PULIZIA DELLA STRINGA PER AST ---
            # Rimuove "tensor(...)"
            dict_str = re.sub(r"tensor\((.*?)\)", r"\1", dict_str)
            # Rimuove "np.float64(...)"
            dict_str = re.sub(r"np\.float64\((.*?)\)", r"\1", dict_str)
            # -------------------------------------

            # Ora la stringa è pulita e ast può leggerla come dizionario
            metrics_dict = ast.literal_eval(dict_str)
            
            f1 = metrics_dict.get('f1', 0)
            prec = metrics_dict.get('precision', 0)
            rec = metrics_dict.get('recall', 0)
            auc = metrics_dict.get('auc', 0)
            acc = metrics_dict.get('accuracy', 0)
            
            print(f"{thresh:<5} | {f1:.4f}   | {prec:.4f}     | {rec:.4f}     | {auc:.4f}   | {acc:.4f}   ")
            
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thresh
        else:
            print(f"{thresh:<5} | ERRORE: Output non parsabile. (Controlla se il training è finito/file esiste)")
            # print(result.stderr) # Decommenta per vedere l'errore tecnico se serve

    except Exception as e:
        print(f"{thresh:<5} | ERRORE ESECUZIONE: {e}")

print("-" * 75)
print(f"MIGLIOR RISULTATO: Threshold {best_thr} -> F1: {best_f1:.4f}")