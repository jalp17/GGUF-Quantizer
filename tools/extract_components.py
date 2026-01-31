import os
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def extract_components(input_path, output_dir):
    print(f"[*] Analizando modelo: {os.path.basename(input_path)}")
    state_dict = load_file(input_path, device="cpu")
    
    components = {
        "unet": {},
        "clip_l": {},
        "clip_g": {},
        "t5xxl": {},
        "vae": {}
    }
    
    # Prefijos comunes y específicos (Ampliados para evitar huérfanos)
    prefixes = {
        "unet": [
            "model.diffusion_model.", 
            "diffusion_model."
        ],
        "clip_l": [
            "cond_stage_model.transformer.", 
            "conditioner.embedders.0.transformer.",
            "cond_stage_model.model.", # SD1.5/2.1
            "cond_stage_model.logit_scale",
            "conditioner.embedders.0.logit_scale",
            "conditioner.embedders.0.pooling_"
        ],
        "clip_g": [
            "conditioner.embedders.1.model.",
            "conditioner.embedders.1.logit_scale",
            "conditioner.embedders.1.pooling_"
        ],
        "t5xxl": [
            "conditioner.embedders.2.transformer.", 
            "conditioner.embedders.1.transformer.",
            "conditioner.embedders.2.logit_scale"
        ],
        "vae": [
            "first_stage_model.", 
            "vae."
        ]
    }

    # Distribución de tensores
    for key, tensor in tqdm(state_dict.items(), desc="Segmentando tensores"):
        found = False
        
        # 1. Búsqueda por prefijo exacto
        for comp_name, comp_prefixes in prefixes.items():
            for p in comp_prefixes:
                if key.startswith(p):
                    components[comp_name][key] = tensor
                    found = True
                    break
            if found: break
        
        # 2. Búsqueda por palabras clave para tensores sueltos (CRÍTICO)
        if not found:
            # Tensores de pooling o logit scales que no empezaban con el prefijo anterior
            if "logit_scale" in key:
                if "embedders.0" in key or "cond_stage" in key: components["clip_l"][key] = tensor
                elif "embedders.1" in key: components["clip_g"][key] = tensor
                found = True
            elif "pooling" in key:
                if "embedders.0" in key or "cond_stage" in key: components["clip_l"][key] = tensor
                elif "embedders.1" in key: components["clip_g"][key] = tensor
                found = True
            # Soporte para modelos sin prefijos (ya extraídos pero incompletos)
            elif "diffusion_model" in key: 
                components["unet"][key] = tensor
                found = True
            elif "encoder" in key and "text" in key: 
                components["clip_l"][key] = tensor
                found = True

    # Guardar componentes encontrados
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for name, sd in components.items():
        if sd:
            out_file = os.path.join(output_dir, f"{name}.safetensors")
            # Para el UNET, queremos que el conversor GGUF lo reconozca
            # Si ya tiene el prefijo de difusión, lo dejamos. Si no, lo empaquetamos
            # como un UNET estándar para que convert.py lo detecte.
            save_file(sd, out_file)
            results[name] = out_file
            print(f"[+] Extraído: {name} ({len(sd)} tensores) -> {out_file}")

    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python extract_components.py <input.safetensors> <output_dir>")
    else:
        extract_components(sys.argv[1], sys.argv[2])
