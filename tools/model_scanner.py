import os
import torch
from safetensors import safe_open
from collections import defaultdict

def scan_model(path, depth=3):
    """
    Escanea un modelo y agrupa sus tensores por jerarquía (prefijos).
    depth: niveles de profundidad para agrupar (ej: 3 -> a.b.c)
    """
    if not os.path.exists(path):
        print(f"❌ Archivo no encontrado: {path}")
        return

    print(f"🔍 Escaneando estructura de: {os.path.basename(path)}")
    hierarchy = defaultdict(lambda: {"count": 0, "size_mb": 0.0})
    total_params = 0
    
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = f.keys()
        for key in keys:
            tensor_slice = f.get_slice(key)
            shape = tensor_slice.get_shape()
            
            # Calcular tamaño aproximado
            params = 1
            for dim in shape: params *= dim
            total_params += params
            
            # Obtener prefijo jerárquico
            parts = key.split('.')
            prefix = ".".join(parts[:depth])
            
            hierarchy[prefix]["count"] += 1
            # Asumiendo FP16 (2 bytes por param)
            hierarchy[prefix]["size_mb"] += (params * 2) / (1024 * 1024)

    print("\n" + "="*60)
    print(f"{'RAMA / PREFIJO':<45} | {'TEORES':<8} | {'TAMAÑO MB'}")
    print("-" * 60)
    
    # Ordenar por tamaño para ver qué es lo más pesado
    sorted_items = sorted(hierarchy.items(), key=lambda x: x[1]["size_mb"], reverse=True)
    
    for prefix, data in sorted_items:
        print(f"{prefix:<45} | {data['count']:<8} | {data['size_mb']:>8.2f} MB")

    print("="*60)
    print(f"Total Tensores: {len(keys)}")
    print(f"Total Parámetros: {total_params / 1e6:.2f}M")
    print(f"Peso total estimado (FP16): {(total_params * 2) / (1024**3):.2f} GB")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python model_scanner.py <modelo.safetensors> [profundidad]")
    else:
        d = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        scan_model(sys.argv[1], depth=d)
