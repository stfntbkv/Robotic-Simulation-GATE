#!/usr/bin/env python3
"""
Simple script to download only the LIBERO-10 model from UniVLA.
"""

from pathlib import Path

def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    
    # Create models directory
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    print("Downloading UniVLA LIBERO-10 model...")
    print("From: https://huggingface.co/qwbu/univla-7b-224-sft-libero/tree/main/univla-libero-10")
    
    # Download only the univla-libero-10 subfolder
    snapshot_download(
        repo_id="qwbu/univla-7b-224-sft-libero",
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
        allow_patterns="univla-libero-10/*"
    )
    
    print(f"✓ Download completed to: {models_dir}/univla-libero-10")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()