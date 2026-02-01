import os
import gc
import torch
import shutil
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

def extract_components(input_path, output_dir):
    print(f"[*] Analizando modelo: {os.path.basename(input_path)}")
    
    try:
        f = safe_open(input_path, framework="pt", device="cpu")
        keys = f.keys()
    except Exception as e:
        print(f"[!] Error al abrir el archivo: {e}")
        raise

    configs = {
        "unet": {"prefixes": ["model.diffusion_model.", "diffusion_model."], "data": {}},
        "clip_l": {
            "prefixes": [
                "cond_stage_model.transformer.", "conditioner.embedders.0.transformer.",
                "cond_stage_model.model.", "conditioner.embedders.0.model."
            ],
            "extra_keys": ["logit_scale", "pooling"],
            "data": {}
        },
        "clip_g": {"prefixes": ["conditioner.embedders.1.model."], "extra_keys": ["logit_scale", "pooling"], "data": {}},
        "t5xxl": {"prefixes": ["conditioner.embedders.2.transformer.", "conditioner.embedders.1.transformer."], "data": {}},
        "vae": {"prefixes": ["first_stage_model.", "vae."], "data": {}}
    }

    os.makedirs(output_dir, exist_ok=True)
    
    # Procesar componentes pequeños directamente en memoria r-r-r-r-am!
    # El UNET lo procesaremos al final o de forma especial si es necesario
    
    print("[*] Segmentando tensores...")
    for key in tqdm(keys):
        tensor = f.get_tensor(key)
        found = False
        
        for name, cfg in configs.items():
            for p in cfg.get("prefixes", []):
                if key.startswith(p):
                    new_key = key[len(p):]
                    if name == "unet":
                        new_key = "model.diffusion_model." + new_key
                    cfg["data"][new_key] = tensor
                    found = True
                    break
            if found: break
            
            if "extra_keys" in cfg:
                for ek in cfg["extra_keys"]:
                    if ek in key:
                        if name == "clip_l" and ("embedders.0" in key or "cond_stage" in key):
                            cfg["data"][key.split('.')[-1]] = tensor
                            found = True
                        elif name == "clip_g" and "embedders.1" in key:
                            cfg["data"][key.split('.')[-1]] = tensor
                            found = True
            if found: break
        
        # Si NO lo encontramos o tras procesar, liberamos si es necesario
        # Aunque aquí el problema es el 'data' dict que crece.
        # Si la RAM es crítica, guardaremos el UNET antes que el resto.

    # Guardar componentes
    for name, cfg in configs.items():
        sd = cfg["data"]
        if sd:
            out_file = os.path.join(output_dir, f"{name}.safetensors")
            print(f"[*] Guardando {name} ({len(sd)} tensores)...")
            save_file(sd, out_file)
            print(f"[+] Extraído: {name} -> {out_file}")
            # Liberar memoria tras guardar cada componente
            cfg["data"] = {}
            gc.collect()

if __name__ == "__main__":
    import sys
    import traceback
    if len(sys.argv) < 3:
        print("Uso: python extract_components.py <input.safetensors> <output_dir>")
    else:
        try:
            extract_components(sys.argv[1], sys.argv[2])
        except Exception as e:
            print(f"\n[!] ERROR DURANTE LA EXTRACCIÓN:")
            traceback.print_exc()
            sys.exit(1)
