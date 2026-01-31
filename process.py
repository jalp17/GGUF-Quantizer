import sys
import os
import argparse
from quantizer.quantizer import GGUFQuantizer

def main():
    parser = argparse.ArgumentParser(description="GGUF Quantizer - Herramienta de automatización para modelos de imagen")
    
    # Argumentos de entrada
    parser.add_argument("links", nargs="*", help="URLs de CivitAI para procesar")
    parser.add_argument("-f", "--file", help="Archivo de texto con una URL por línea")
    
    # Parámetros de proceso
    parser.add_argument("--no-upload", action="store_true", help="No subir archivos a Hugging Face (solo proceso local)")
    parser.add_argument("-q", "--quants", help="Niveles de cuantización separados por comas (ej: Q4_K_M,Q8_0)")
    
    args = parser.parse_args()
    
    # Recopilar enlaces
    all_links = args.links if args.links else []
    
    if args.file:
        if os.path.exists(args.file):
            with open(args.file, "r") as f:
                file_links = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                all_links.extend(file_links)
        else:
            print(f"❌ Error: El archivo {args.file} no existe.")
            sys.exit(1)
            
    if not all_links:
        print("❌ Error: No se proporcionaron URLs ni archivo de entrada.")
        parser.print_help()
        sys.exit(1)

    # Configurar cuantización
    custom_quants = None
    if args.quants:
        custom_quants = [q.strip() for q in args.quants.split(",")]
        print(f"⚖️ Usando niveles de cuantización personalizados: {custom_quants}")

    # Iniciar Quantizer
    quantizer = GGUFQuantizer()
    upload_enabled = not args.no_upload
    
    print(f"🚀 Iniciando proceso para {len(all_links)} modelos...")
    if not upload_enabled:
        print("⚠️ Modo LOCAL-ONLY activado. Los archivos no se subirán a Hugging Face.")

    for link in all_links:
        try:
            quantizer.process(link, upload_to_hf=upload_enabled, custom_quants=custom_quants)
        except Exception as e:
            print(f"❌ Falló procesamiento de {link}: {e}")

if __name__ == "__main__":
    main()
