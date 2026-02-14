
import os
import sys

print(f"DEBUG: sys.executable = {sys.executable}")
print(f"DEBUG: sys.path = {sys.path}")

import torch
import subprocess
from safetensors.torch import save_file

def create_dummy_model(path):
    print(f"Creating dummy model at {path}...")
    tensors = {
        "model.diffusion_model.time_embed.0.weight": torch.randn(320, 1280),
        "model.diffusion_model.time_embed.0.bias": torch.randn(320),
        "model.diffusion_model.input_blocks.1.1.proj_in.weight": torch.randn(320, 320, 1, 1),
        # Add a tensor that triggers reshape logic (simulate flux/sdxl)
        # 5 * 256 = 1280 elements
        "model.diffusion_model.reshape_test.weight": torch.randn(5, 256), 
    }
    save_file(tensors, path)
    print("Dummy model created.")

def test():
    dummy_safetensors = "dummy_model.safetensors"
    output_gguf = "dummy_model.gguf"
    
    create_dummy_model(dummy_safetensors)
    
    convert_script = os.path.join("tools", "convert.py")
    
    # Run with --low-ram
    print("\nRunning convert.py --low-ram...")
    cmd = [sys.executable, convert_script, "--src", dummy_safetensors, "--dst", output_gguf, "--low-ram"]
    subprocess.run(cmd, check=True)
    
    if os.path.exists(output_gguf):
        print(f"\nSUCCESS: {output_gguf} created using --low-ram mode.")
        print(f"Size: {os.path.getsize(output_gguf)} bytes")
    else:
        print("\nFAILURE: Output file not found.")
        sys.exit(1)

    # Cleanup
    # os.remove(dummy_safetensors)
    # os.remove(output_gguf)

if __name__ == "__main__":
    test()
