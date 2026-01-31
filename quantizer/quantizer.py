import os
import sys
import subprocess
import shutil
import requests
import re
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

# Cargar variables de entorno desde .env si existe
load_dotenv()

def make_safe_name(name):
    """Limpia el nombre para cumplir con las reglas de Hugging Face (Repo ID)."""
    name = name.lower()
    name = re.sub(r'[^a-z0-9._-]', '-', name)
    name = re.sub(r'[-._]{2,}', '-', name)
    return name.strip('-._')[:80]

# ==========================================
# CONFIGURACIÓN LOCAL
# ==========================================
class Config:
    HF_TOKEN = os.getenv('HF_TOKEN')
    HF_USER = os.getenv('HF_USER')
    CIVITAI_API_KEY = os.getenv('CIVITAI_API_KEY')

    # Niveles de cuantización deseados
    QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0"]

    # Directorios de trabajo locales
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(ROOT_DIR, "input")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    
    # Rutas de herramientas (asume que llama.cpp está compilado en la raíz o subcarpeta)
    # Ajustar según SO (Windows usa .exe)
    QUANTIZE_BIN = os.path.join(ROOT_DIR, "llama.cpp", "build", "bin", "llama-quantize")
    if os.name == 'nt' and not QUANTIZE_BIN.endswith(".exe"):
        QUANTIZE_BIN += ".exe"
        
    CONVERT_SCRIPT = os.path.join(ROOT_DIR, "tools", "convert.py")

# ==========================================
# SERVICIO DE METADATOS (CivitAI)
# ==========================================
class CivitaiClient:
    @staticmethod
    def get_metadata(url):
        try:
            parts = url.strip().split('/')
            model_id = next((parts[i+1] for i, p in enumerate(parts) if p == 'models'), None)
            if not model_id and 'modelVersionId=' in url:
                model_id = url.split('modelVersionId=')[1].split('&')[0]
            if not model_id: return None

            headers = {"Authorization": f"Bearer {Config.CIVITAI_API_KEY}"} if Config.CIVITAI_API_KEY else {}
            resp = requests.get(f"https://civitai.com/api/v1/models/{model_id}", headers=headers)
            if resp.status_code != 200: return None

            data = resp.json()
            ver = data['modelVersions'][0]
            
            return {
                "id": model_id,
                "name": data['name'],
                "author": data.get('creator', {}).get('username', 'Unknown'),
                "download_url": f"{ver['downloadUrl']}?token={Config.CIVITAI_API_KEY}" if Config.CIVITAI_API_KEY else ver['downloadUrl'],
                "description": data.get("description", ""),
                "baseModel": ver.get("baseModel", "SDXL")
            }
        except Exception: return None

# ==========================================
# GENERADOR DE DOCUMENTACIÓN (PREMIUM)
# ==========================================
class Documentation:
    @staticmethod
    def clean_html(raw_html):
        if not raw_html: return ""
        html = raw_html.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
        html = html.replace('</p>', '\n\n').replace('</div>', '\n').replace('</li>', '\n')
        clean_text = re.sub(r'<.*?>', '', html)
        clean_text = clean_text.replace('&nbsp;', ' ').replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
        return clean_text.strip()

    @staticmethod
    def generate_readme(meta, quant_list):
        arch = meta['baseModel']
        if "illustrious" in arch.lower(): arch = "Illustrious"
        elif "pony" in arch.lower(): arch = "Pony"
        
        yaml = f"---\nlicense: other\ntags:\n- stable-diffusion\n- {arch.lower()}\n- gguf\n- comfyui\n---\n"
        
        quants_table = "| Versión | Tipo | Uso Recomendado | Calidad |\n| :--- | :--- | :--- | :--- |\n"
        for q in quant_list:
            q_type = "Medium" if "Q4" in q else "Large" if "Q5" in q else "Ultra"
            stars = "⭐⭐⭐" if "Q4" in q else "⭐⭐⭐⭐" if "Q5" in q else "⭐⭐⭐⭐⭐"
            usage = "Balance perfecto" if "Q4" in q else "Calidad superior"
            quants_table += f"| **{q}** | {q_type} | {usage} | {stars} |\n"

        clean_desc = Documentation.clean_html(meta.get('description', ""))
        
        return yaml + f"""# {meta['name']} - GGUF Collection 🚀

Este repositorio contiene versiones optimizadas en formato **GGUF** del modelo original [{meta['name']}](https://civitai.com/models/{meta['id']}).

---

## 👤 Créditos y Autoría
- **Autor Original:** [{meta['author']}](https://civitai.com/user/{meta['author']})
- **Proceso de Cuantización:** [{Config.HF_USER}](https://huggingface.co/{Config.HF_USER})

> [!IMPORTANT]
> **Aviso de Autoría:** Este modelo no es de mi creación. Yo únicamente he realizado la cuantización (compresión) técnica. Todo el mérito creativo pertenece al autor original mencionado arriba.

---

## 📊 Tabla de Versiones y Recomendaciones
{quants_table}

---

## ⚙️ Detalles Técnicos
- **Arquitectura Base:** {arch}
- **Formato:** GGUF (Agnóstico)
- **Herramientas:** llama.cpp & ComfyUI-GGUF

---

## 🚀 Instrucciones para ComfyUI
1. **GGUF (UNET)**: Coloque los archivos `.gguf` en `ComfyUI/models/unet/`
2. **Componentes**: CLIP/VAE en sus carpetas respectivas.

---

## 📝 Descripción del Autor Original
{clean_desc}

---
*Generado por GGUF Quantizer Modular*
"""

# GGUF Quantizer Modular
class GGUFQuantizer:
    def __init__(self):
        self.api = HfApi()
        os.makedirs(Config.INPUT_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    def download_file(self, url, dest):
        print(f"📥 Descargando a: {dest}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    def update_readme(self, url, custom_quants=None):
        print(f"📝 Actualizando solo documentación para: {url}")
        meta = CivitaiClient.get_metadata(url)
        if not meta: return
        
        safe_name = make_safe_name(meta['name'])
        repo_id = f"{Config.HF_USER}/{safe_name}-GGUF"
        quants = custom_quants if custom_quants else Config.QUANTS
        
        readme = Documentation.generate_readme(meta, quants)
        self.api.upload_file(
            path_or_fileobj=readme.encode("utf-8"), 
            path_in_repo="README.md", 
            repo_id=repo_id, 
            token=Config.HF_TOKEN
        )
        print(f"✅ README actualizado en: {repo_id}")

    def process(self, url, upload_to_hf=True, custom_quants=None):
        print(f"\n🌟 Procesando: {url}")
        meta = CivitaiClient.get_metadata(url)
        if not meta: return
        
        safe_name = make_safe_name(meta['name'])
        repo_id = f"{Config.HF_USER}/{safe_name}-GGUF"
        
        # Determinar quants a usar
        quants_to_process = custom_quants if custom_quants else Config.QUANTS
        
        # 1. Descarga
        raw_model = os.path.join(Config.INPUT_DIR, f"{safe_name}.safetensors")
        self.download_file(meta['download_url'], raw_model)
        
        # 2. Extracción
        from tools.extract_components import extract_components
        comp_dir = os.path.join(Config.INPUT_DIR, f"components_{safe_name}")
        extracted = extract_components(raw_model, comp_dir)
        
        # 3. Subir Componentes No-Unet (Si procede)
        if upload_to_hf:
            print(f"📦 Creando/Verificando repo: {repo_id}")
            create_repo(repo_id, token=Config.HF_TOKEN, exist_ok=True)
            
            for name, path in extracted.items():
                if name != "unet":
                    print(f"⬆️ Subiendo {name}...")
                    self.api.upload_file(path_or_fileobj=path, path_in_repo=f"{name}.safetensors", repo_id=repo_id, token=Config.HF_TOKEN)

        # 4. FP16 GGUF Base
        unet_path = extracted.get("unet")
        fp16_path = os.path.join(Config.OUTPUT_DIR, f"{safe_name}.fp16.gguf")
        print("⚙️ Generando base FP16...")
        subprocess.run([sys.executable, Config.CONVERT_SCRIPT, "--src", unet_path, "--dst", fp16_path], check=True)
        
        # 5. Cuantización
        for q in quants_to_process:
            print(f"⚖️ Cuantizando a {q}...")
            quant_name = f"{safe_name}.{q}.gguf"
            quant_path = os.path.join(Config.OUTPUT_DIR, quant_name)
            subprocess.run([Config.QUANTIZE_BIN, fp16_path, quant_path, q], check=True)
            
            if upload_to_hf:
                print(f"⬆️ Subiendo {quant_name}...")
                self.api.upload_file(path_or_fileobj=quant_path, path_in_repo=quant_name, repo_id=repo_id, token=Config.HF_TOKEN)
            else:
                print(f"💾 Guardado localmente: {quant_path}")
            
            # Nota: Si no subimos, quizás el usuario quiera conservar el archivo. 
            # Si subimos, lo borramos para ahorrar espacio. Por ahora mantenemos el borrado tras subida.
            if upload_to_hf:
                os.remove(quant_path)

        # 6. README (Si procede)
        if upload_to_hf:
            print("📝 Actualizando documentación en HF...")
            readme = Documentation.generate_readme(meta, quants_to_process)
            self.api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id, token=Config.HF_TOKEN)

        # Limpieza
        os.remove(raw_model)
        os.remove(fp16_path)
        shutil.rmtree(comp_dir)
        print(f"✅ Finalizado procesamiento de: {meta['name']}")
