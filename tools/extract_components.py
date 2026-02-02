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
    
    # 1. Mapeo de claves por componente (sin cargar tensores todavía)
    keys_map = {name: {} for name in configs.keys()}
    print("[*] Mapeando estructura del modelo...")
    for key in tqdm(keys):
        found = False
        for name, cfg in configs.items():
            for p in cfg.get("prefixes", []):
                if key.startswith(p):
                    new_key = key[len(p):]
                    if name == "unet":
                        new_key = "model.diffusion_model." + new_key
                    keys_map[name][new_key] = key
                    found = True
                    break
            if found: break
            
            if "extra_keys" in cfg:
                for ek in cfg["extra_keys"]:
                    if ek in key:
                        if name == "clip_l" and ("embedders.0" in key or "cond_stage" in key):
                            keys_map[name][key.split('.')[-1]] = key
                            found = True
                        elif name == "clip_g" and "embedders.1" in key:
                            keys_map[name][key.split('.')[-1]] = key
                            found = True
            if found: break

    # 2. Extracción secuencial (Componente por Componente) para ahorrar RAM
    extracted_paths = {}
    for name, m_keys in keys_map.items():
        if not m_keys: continue
        
        print(f"[*] Procesando componente: {name} ({len(m_keys)} tensores)...")
        component_data = {}
        
        # Extraer solo los tensores de este componente
        for new_key, original_key in m_keys.items():
            tensor = f.get_tensor(original_key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            component_data[new_key] = tensor
            
        # Guardar componente inmediatamente
        out_file = os.path.join(output_dir, f"{name}.safetensors")
        save_file(component_data, out_file)
        print(f"[+] Extraído: {name} -> {out_file}")
        extracted_paths[name] = out_file
        
        # Liberación agresiva de RAM
        component_data.clear()
        del component_data
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return extracted_paths
    
    return extracted_paths

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
