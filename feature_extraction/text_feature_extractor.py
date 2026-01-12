import argparse
import json
import os
import glob
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add current path to import core modules
sys.path.append(os.getcwd())

try:
    from core.vision_encoder import pe
    from core.vision_encoder.tokenizer import SimpleTokenizer
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    print("Please run this script from INSIDE the 'feature_extraction' folder.")
    sys.exit(1)

def load_graph_json(json_path):
    """Loads the JSON file and returns the 'steps' dictionary."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['steps']

def clean_step_text(text):
    """
    Cleans the step text description.
    Example: "Add-1/2 tsp..." -> "Add 1/2 tsp..."
    This couse PE is trained on real sentences so Add-1/2 is viewed as a token instead of being considered as 2 words, so in this way it perform better
    """
    return text.replace("-", " ").strip()

def process_single_graph(json_path, model, tokenizer, context_length, device):
    """
    Extracts text features for a single JSON graph file and returns a dictionary.
    """
    steps = load_graph_json(json_path)
    features_dict = {}

    for step_id, raw_text in steps.items():
        # Text cleaning
        if raw_text == "START" or raw_text == "END":
            clean_text = raw_text
        else:
            clean_text = clean_step_text(raw_text)
        
        # Manual tokenization (following standard CLIP implementation)
        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + tokenizer.encode(clean_text) + [eot_token]
        
        # Padding / Truncating
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            tokens[-1] = eot_token
        else:
            tokens += [0] * (context_length - len(tokens))
        
        text_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

        # Feature Extraction
        with torch.no_grad():
            # encode_text returns normalized features if normalize=True
            text_features = model.encode_text(text_tensor, normalize=True)
        
        # Save as Numpy array (CPU)
        features_dict[step_id] = text_features.cpu().numpy()
        
    return features_dict

def main():
    parser = argparse.ArgumentParser(description="Extract text features from graph JSON files using Perception Encoder.")
    parser.add_argument("--graphs_dir", type=str, required=True, help="Directory containing the graph JSON files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where .npz files will be saved.")
    parser.add_argument("--model", type=str, default="PE-Core-B16-224", help="Model config name (must match the one used for video features).")
    args = parser.parse_args()

    # Handle relative paths safely
    if args.graphs_dir.startswith(".."):
        base_dir = os.path.dirname(os.getcwd())
        args.graphs_dir = os.path.join(base_dir, args.graphs_dir.strip("../"))
        args.output_dir = os.path.join(base_dir, args.output_dir.strip("../"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load the Model (Once)
    print(f"Loading model: {args.model}...")
    model = pe.CLIP.from_config(args.model, pretrained=True)
    model.to(device)
    model.eval()

    # 2. Load the Tokenizer
    tokenizer = SimpleTokenizer()
    context_length = model.context_length
    print(f"Model Context Length: {context_length}")

    # 3. Find all JSON files
    json_files = glob.glob(os.path.join(args.graphs_dir, "*.json"))
    print(f"Found {len(json_files)} graph files in '{args.graphs_dir}'. Starting processing...")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Process loop
    for json_path in tqdm(json_files, desc="Processing Graphs"):
        try:
            # Extract features
            features_dict = process_single_graph(json_path, model, tokenizer, context_length, device)
            
            # Determine output filename
            # Example: "code/graphs/recipe.json" -> "recipe.npz"
            filename = os.path.basename(json_path).replace(".json", ".npz")
            output_path = os.path.join(args.output_dir, filename)
            
            # Save compressed .npz
            np.savez_compressed(output_path, **features_dict)
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

    print("Extraction completed for all graphs!")

if __name__ == "__main__":
    main()