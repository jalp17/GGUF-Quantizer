import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def extract_components(input_path, output_dir):
    print(f"[*] Analizando modelo: {os.path.basename(input_path)}")
    state_dict = load_file(input_path, device="cpu")
    
    # Definición de componentes y sus prefijos a ELIMINAR para el archivo final
    configs = {
        "unet": {
            "prefixes": ["model.diffusion_model.", "diffusion_model."],
            "keep_prefix": False, # convert.py suele preferir sin prefijo o con un prefijo específico que él añade
            "data": {}
        },
        "clip_l": {
            "prefixes": [
                "cond_stage_model.transformer.", 
                "conditioner.embedders.0.transformer.",
                "cond_stage_model.model.",
                "conditioner.embedders.0.model."
            ],
            "extra_keys": ["logit_scale", "pooling"],
            "data": {}
        },
        "clip_g": {
            "prefixes": ["conditioner.embedders.1.model."],
            "extra_keys": ["logit_scale", "pooling"],
            "data": {}
        },
        "t5xxl": {
            "prefixes": ["conditioner.embedders.2.transformer.", "conditioner.embedders.1.transformer."],
            "data": {}
        },
        "vae": {
            "prefixes": ["first_stage_model.", "vae."],
            "data": {}
        }
    }

    # Distribución y limpieza de tensores
    for key, tensor in tqdm(state_dict.items(), desc="Segmentando y limpiando tensores"):
        found = False
        
        for name, cfg in configs.items():
            for p in cfg.get("prefixes", []):
                if key.startswith(p):
                    # Limpiar el nombre del tensor (quitar el prefijo)
                    new_key = key[len(p):]
                    
                    # Caso especial UNET: convert.py a veces necesita el prefijo diffusion_model.
                    # pero si lo extraemos para ComfyUI nativo (no GGUF), suele ser limpio.
                    # Para nuestro flujo GGUF, lo dejaremos limpio y que convert.py lo maneje.
                    if name == "unet":
                        # Asegurar que el unet tenga el formato que espera convert.py
                        # Si convert.py de ComfyUI-GGUF espera 'model.diffusion_model.', se lo ponemos.
                        new_key = "model.diffusion_model." + new_key
                    
                    cfg["data"][new_key] = tensor
                    found = True
                    break
            if found: break
            
            # Manejo de llaves extra (logit_scale, etc) que no están en el prefijo principal
            if "extra_keys" in cfg:
                for ek in cfg["extra_keys"]:
                    if ek in key:
                        # Identificar a qué clip pertenece por el índice del embedder
                        if name == "clip_l" and ("embedders.0" in key or "cond_stage" in key):
                            cfg["data"][key.split('.')[-1]] = tensor # Guardar solo la parte final (ej: logit_scale)
                            found = True
                        elif name == "clip_g" and "embedders.1" in key:
                            cfg["data"][key.split('.')[-1]] = tensor
                            found = True
            if found: break

    # Guardar componentes encontrados
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for name, cfg in configs.items():
        sd = cfg["data"]
        if sd:
            # Corrección de nombres para CLIP_L / CLIP_G para que ComfyUI los trague bien
            if name.startswith("clip"):
                # Si no tiene el prefijo de transformer, a veces ayuda añadirlo o quitarlo
                # pero lo más seguro es dejarlo como está tras el strip de arriba.
                pass
            
            out_file = os.path.join(output_dir, f"{name}.safetensors")
            save_file(sd, out_file)
            results[name] = out_file
            print(f"[+] Extraído y Limpiado: {name} ({len(sd)} tensores) -> {out_file}")

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python extract_components.py <input.safetensors> <output_dir>")
    else:
        extract_components(sys.argv[1], sys.argv[2])
