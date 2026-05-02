#!/usr/bin/env python3
"""
Setup FLUX.2-klein model directory from local files.

Usage:
    python setup-model-dir.py MODEL [--output-dir DIR] [--copy]

This script helps you set up the model directory structure needed for iris
by using local files you already have, instead of downloading them.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

try:
    import readline
    def path_completer(text, state):
        import glob
        text = os.path.expanduser(text)
        matches = glob.glob(text + '*')
        if state < len(matches):
            m = matches[state]
            if os.path.isdir(m) and not m.endswith('/'):
                return m + '/'
            return m
        return None
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(path_completer)
except ImportError:
    pass

MODELS = {
    "4b": ("black-forest-labs/FLUX.2-klein-4B", "./flux-klein-4b"),
    "4b-base": ("black-forest-labs/FLUX.2-klein-base-4B", "./flux-klein-4b-base"),
    "9b": ("black-forest-labs/FLUX.2-klein-9B", "./flux-klein-9b"),
    "9b-base": ("black-forest-labs/FLUX.2-klein-base-9B", "./flux-klein-9b-base"),
    "zimage-turbo": ("Tongyi-MAI/Z-Image-Turbo", "./zimage-turbo"),
}

def ask_path(prompt, default=None, must_exist=True, is_dir=False):
    while True:
        try:
            val = input(f"{prompt} [{default if default else ''}]: ").strip()
            if not val and default:
                val = default
            if not val:
                print("Error: path is required")
                continue
            
            path = Path(val).expanduser().resolve()
            if must_exist and not path.exists():
                print(f"Error: path does not exist: {path}")
                continue
            if must_exist:
                if is_dir and not path.is_dir():
                    print(f"Error: not a directory: {path}")
                    continue
                if not is_dir and not path.is_file():
                    print(f"Error: not a file: {path}")
                    continue
            return path
        except EOFError:
            print("\nAborted.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(1)

def link_or_copy(src, dst, copy=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.readlink() == src:
            return
        if dst.is_file() and not dst.is_symlink():
             # Check if it's the same file (same inode or content)
             if src.stat().st_ino == dst.stat().st_ino:
                 return
        
        print(f"Warning: {dst} already exists, skipping.")
        return

    if copy:
        print(f"Copying {src} -> {dst}")
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        print(f"Linking {src} -> {dst}")
        os.symlink(src, dst)

def main():
    parser = argparse.ArgumentParser(
        description='Setup FLUX.2-klein model directory from local files'
    )
    parser.add_argument(
        'model',
        nargs='?',
        choices=list(MODELS.keys()),
        help='Model type (4b, 4b-base, 9b, 9b-base, zimage-turbo)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory (default: auto based on model type)'
    )
    parser.add_argument(
        '--copy', '-c',
        action='store_true',
        help='Copy files instead of symlinking'
    )
    args = parser.parse_args()

    if not args.model:
        print("Available models:")
        for m in MODELS:
            print(f"  {m}")
        print()
        args.model = input("Select model: ").strip()
        if args.model not in MODELS:
            print(f"Error: invalid model: {args.model}")
            return 1

    repo_id, default_dir = MODELS[args.model]
    output_dir = Path(args.output_dir if args.output_dir else default_dir)

    print(f"Setting up model: {args.model}")
    print(f"Output directory: {output_dir}")
    print()

    # Strategy: 
    # 1. Ask for a base directory that might contain everything.
    # 2. For each missing part, ask specifically.

    base_dir = None
    if input("Do you have a base directory containing all model files? (y/n) [n]: ").lower().startswith('y'):
        base_dir = ask_path("Base directory", is_dir=True)

    # 1. model_index.json
    mi_path = output_dir / "model_index.json"
    mi_src = None
    if base_dir:
        candidate = base_dir / "model_index.json"
        if candidate.exists():
            if input(f"Found existing model_index.json at {candidate}. Use it? (y/n) [y]: ").lower() != 'n':
                mi_src = candidate

    if mi_src:
        link_or_copy(mi_src, mi_path, args.copy)
    else:
        print("Creating model_index.json...")
        print("Select Pipeline Type:")
        print("  1. FluxPipeline (FLUX.2-klein)")
        print("  2. ZImagePipeline (Z-Image-Turbo)")
        
        while True:
            choice = input("Choice [1]: ").strip() or "1"
            if choice == "1":
                class_name = "FluxPipeline"
                break
            elif choice == "2":
                class_name = "ZImagePipeline"
                break
            else:
                print("Invalid choice.")

        is_distilled = input("Is this a distilled/turbo model? (y/n) [y]: ").lower() != 'n'
        
        import json
        mi_content = {
            "_class_name": class_name,
            "is_distilled": is_distilled
        }
        
        print(f"Generating {mi_path}...")
        mi_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mi_path, "w") as f:
            json.dump(mi_content, f, indent=2)
        print("Done.")

    # 2. VAE
    vae_dir_src = None
    if base_dir:
        candidate = base_dir / "vae"
        if candidate.is_dir():
            vae_dir_src = candidate
    
    vae_path = output_dir / "vae"
    vae_cfg_path = vae_path / "config.json"
    vae_st_path = vae_path / "diffusion_pytorch_model.safetensors"

    if vae_dir_src:
        for f in ["diffusion_pytorch_model.safetensors", "config.json"]:
            src_f = vae_dir_src / f
            if src_f.exists():
                link_or_copy(src_f, vae_path / f, args.copy)
            else:
                print(f"Warning: {f} not found in {vae_dir_src}")

    if not vae_st_path.exists():
        vae_st_src = ask_path("Path to vae/diffusion_pytorch_model.safetensors")
        link_or_copy(vae_st_src, vae_st_path, args.copy)

    if not vae_cfg_path.exists():
        print("vae/config.json is missing.")
        if input("Create it using presets for this model? (y/n) [y]: ").lower() != 'n':
            # Presets
            if "zimage" in args.model:
                channels = 16
                scaling = 0.3611
                shift = 0.1159
            else: # Flux
                channels = 16 # Flux-klein typically uses 16 or 32
                scaling = 0.3611
                shift = 0.1159
                print("Note: Flux models usually use 16/32 channels and specific scaling.")
            
            channels = int(input(f"  Latent channels [{channels}]: ").strip() or str(channels))
            scaling = float(input(f"  Scaling factor [{scaling}]: ").strip() or str(scaling))
            shift = float(input(f"  Shift factor [{shift}]: ").strip() or str(shift))

            import json
            vae_cfg_content = {
                "latent_channels": channels,
                "scaling_factor": scaling,
                "shift_factor": shift
            }
            print(f"Generating {vae_cfg_path}...")
            vae_path.mkdir(parents=True, exist_ok=True)
            with open(vae_cfg_path, "w") as f:
                json.dump(vae_cfg_content, f, indent=2)
            print("Done.")
        else:
             vae_cfg_src = ask_path("Path to vae/config.json")
             link_or_copy(vae_cfg_src, vae_cfg_path, args.copy)

    # 3. Transformer
    tf_dir_src = None
    if base_dir:
        candidate = base_dir / "transformer"
        if candidate.is_dir():
            tf_dir_src = candidate
    
    if not tf_dir_src:
        if input("Do you have a 'transformer' directory? (y/n) [y]: ").lower() != 'n':
            tf_dir_src = ask_path("Transformer directory", is_dir=True)
        else:
            # This is complex due to sharding, but let's try
            tf_st = ask_path("Path to transformer/diffusion_pytorch_model.safetensors (or a shard/index)")
            tf_cfg = ask_path("Path to transformer/config.json")
            link_or_copy(tf_cfg, output_dir / "transformer" / "config.json", args.copy)
            if tf_st.name.endswith(".json"): # Index
                link_or_copy(tf_st, output_dir / "transformer" / tf_st.name, args.copy)
                print("Note: You may need to manually link shards referenced in the index.")
            else:
                link_or_copy(tf_st, output_dir / "transformer" / "diffusion_pytorch_model.safetensors", args.copy)

    if tf_dir_src:
        # Link everything in transformer dir that matches patterns
        for f in tf_dir_src.glob("*"):
            if f.is_file() and not f.name.endswith((".bin", ".pt", ".pth")):
                link_or_copy(f, output_dir / "transformer" / f.name, args.copy)

    # 4. Text Encoder
    te_dir_src = None
    if base_dir:
        candidate = base_dir / "text_encoder"
        if candidate.is_dir():
            te_dir_src = candidate
    
    if not te_dir_src:
        te_dir_src = ask_path("Text encoder directory", is_dir=True)
    
    if te_dir_src:
        for f in te_dir_src.glob("*"):
            if f.is_file() and not f.name.endswith((".bin", ".pt", ".pth")):
                link_or_copy(f, output_dir / "text_encoder" / f.name, args.copy)

    # 5. Tokenizer
    tok_dir_src = None
    if base_dir:
        candidate = base_dir / "tokenizer"
        if candidate.is_dir():
            tok_dir_src = candidate
    
    if not tok_dir_src:
        tok_dir_src = ask_path("Tokenizer directory", is_dir=True)
    
    if tok_dir_src:
        for f in tok_dir_src.glob("*"):
            if f.is_file():
                link_or_copy(f, output_dir / "tokenizer" / f.name, args.copy)

    print()
    print("Setup complete!")
    print(f"Model directory: {output_dir}")
    print()
    print("Usage:")
    print(f"  ./flux -d {output_dir} -p \"your prompt\" -o output.png")
    return 0

if __name__ == '__main__':
    sys.exit(main())
