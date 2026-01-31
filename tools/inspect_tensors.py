import os
import torch
from safetensors.torch import load_file
from tqdm import tqdm

def inspect_tensors(original_path, component_dir):
    print(f"[*] Inspeccionando modelo original: {original_path}")
    original_sd = load_file(original_path, device="cpu")
    original_keys = set(original_sd.keys())
    
    components = [f for f in os.listdir(component_dir) if f.endswith(".safetensors")]
    extracted_keys = set()
    
    print(f"[*] Analizando {len(components)} componentes en {component_dir}...")
    for comp in components:
        comp_path = os.path.join(component_dir, comp)
        sd = load_file(comp_path, device="cpu")
        extracted_keys.update(sd.keys())
        print(f"  - {comp}: {len(sd)} tensores")

    orphan_keys = original_keys - extracted_keys
    
    print("\n" + "="*50)
    print("📊 RESULTADOS DE LA INSPECCIÓN")
    print("="*50)
    print(f"Tensores totales en original: {len(original_keys)}")
    print(f"Tensores totales extraídos: {len(extracted_keys)}")
    print(f"Tensores HUÉRFANOS (perplejidad/error): {len(orphan_keys)}")
    
    if orphan_keys:
        print("\n⚠️ TENSORES CRÍTICOS PERDIDOS (Primeros 20):")
        for k in sorted(list(orphan_keys))[:20]:
            print(f"  [MISSING] {k}")
            
        # Alerta específica para CLIP
        clip_orphans = [k for k in orphan_keys if "cond_stage" in k or "conditioner" in k]
        if clip_orphans:
            print(f"\n❌ SE DETECTARON {len(clip_orphans)} TENSORES DE CLIP PERDIDOS. " 
                  "Esto causará imágenes negras o ruido incoherente.")

    else:
        print("\n✅ Integridad total: Todos los tensores fueron asignados.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python inspect_tensors.py <original.safetensors> <component_dir>")
    else:
        inspect_tensors(sys.argv[1], sys.argv[2])
