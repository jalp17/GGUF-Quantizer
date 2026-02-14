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
    # Rutas de herramientas
    QUANTIZE_BIN = os.getenv('QUANTIZE_BIN', os.path.join(ROOT_DIR, "llama.cpp", "build", "bin", "llama-quantize"))
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
            url = url.strip()
            model_id = None
            version_id = None

            # 1. Extraer IDs de la URL mediante Regex
            # Formato: /models/1234
            model_match = re.search(r'/models/(\d+)', url)
            if model_match: model_id = model_match.group(1)

            # Formato: modelVersionId=5678 o /models/1234?modelVersionId=5678
            version_match = re.search(r'modelVersionId=(\d+)', url)
            if version_match: version_id = version_match.group(1)

            # Formato: /api/download/models/5678 (Descarga directa)
            api_match = re.search(r'/api/download/models/(\d+)', url)
            if api_match: version_id = api_match.group(1)

            if not model_id and not version_id:
                print(f"⚠️ No se pudo extraer ID de la URL: {url}")
                return None

            headers = {"Authorization": f"Bearer {Config.CIVITAI_API_KEY}"} if Config.CIVITAI_API_KEY else {}
            
            # 2. Si tenemos Version ID, resolvemos sus metadatos específicos primero
            version_data = None
            if version_id:
                resp_v = requests.get(f"https://civitai.com/api/v1/model-versions/{version_id}", headers=headers)
                if resp_v.status_code == 200:
                    version_data = resp_v.json()
                    model_id = str(version_data['modelId']) # Asegurar que tenemos el model_id correcto
                else:
                    print(f"⚠️ Error al obtener versión {version_id}: {resp_v.status_code}")
                    if not model_id: return None

            # 3. Obtener metadatos globales del modelo
            resp_m = requests.get(f"https://civitai.com/api/v1/models/{model_id}", headers=headers)
            if resp_m.status_code != 200:
                print(f"⚠️ Error al obtener modelo {model_id}: {resp_m.status_code}")
                return None
            
            model_data = resp_m.json()
            
            # 4. Seleccionar la versión correcta (priorizar la solicitada o la última)
            if version_data:
                ver = version_data
            else:
                # Si no se especificó versión, usar la primera (la más reciente)
                ver = model_data['modelVersions'][0]
                version_id = str(ver['id'])

            return {
                "id": model_id,
                "version_id": version_id,
                "name": model_data['name'],
                "version_name": ver.get('name', 'Main'),
                "author": model_data.get('creator', {}).get('username', 'Unknown'),
                "download_url": f"{ver['downloadUrl']}?token={Config.CIVITAI_API_KEY}" if Config.CIVITAI_API_KEY else ver['downloadUrl'],
                "description": model_data.get("description", ""),
                "baseModel": ver.get("baseModel", "SDXL")
            }
        except Exception as e:
            print(f"❌ Error en CivitaiClient: {e}")
            return None

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
        
        quants_table = "| Versión | Tipo | Peso | Calidad | Uso Recomendado |\n| :--- | :--- | :--- | :--- | :--- |\n"
        for q in quant_list:
            q_type = "Medium" if "Q4" in q else "Large" if "Q5" in q else "Ultra"
            stars = "Basica" if "Q4" in q else "Alta" if "Q5" in q else "Alta"
            usage = "Balance / Velocidad" if "Q4" in q else "Uso Profesional"
            size = "~5-7GB" if "Q4" in q else "~8-10GB" if "Q5" in q else "Original"
            quants_table += f"| **{q}** | {q_type} | {size} | {stars} | {usage} |\n"

        clean_desc = Documentation.clean_html(meta.get('description', ""))
        
        return yaml + f"""# {meta['name']} - GGUF Ultimate Edition 🏭

Este repositorio ofrece la colección definitiva en formato **GGUF** del modelo original [{meta['name']}](https://civitai.com/models/{meta['id']}). Optimizados para un rendimiento máximo en **ComfyUI**.

---

## � Tabla de Comparativa de Cuantizaciones
{quants_table}

---

## ⚙️ Guía de Optimización y Comparativa de Parámetros

Para obtener los mejores resultados con esta versión GGUF, se recomiendan los siguientes ajustes:

### 1. 🎚️ Comparativa de CFG (Classifier Free Guidance)
| CFG Scale | Efecto en GGUF | Resultado Visual |
| :--- | :--- | :--- |
| **1.0 - 3.5** | Suave / Realista | Menos contraste, ideal para estilos fotográficos. |
| **4.0 - 6.5** | **Recomendado** | Balance perfecto entre fidelidad al prompt y detalle. |
| **7.0 - 9.0** | Estilizado | Colores más saturados y bordes más definidos. |

### 2. ⚡ Comparativa de Pasos (Sampling Steps)
| Pasos | Rendimiento | Nivel de Detalle |
| :--- | :--- | :--- |
| **15 - 20** | Ultra Rápido | Bocetos rápidos o previsualizaciones. |
| **25 - 35** | **Óptimo** | El "Sweet Spot" para GGUF con casi cero ruido. |
| **40+** | Estándar | Máximo refinamiento de texturas complejas. |

### 3. 🌫️ Comparativa de Denoise (Solo para i2i / Hires Fix)
*   **0.35 - 0.45**: Mantiene la estructura original pero con limpieza de artefactos.
*   **0.50 - 0.65**: El rango ideal para añadir detalle sin deformar el sujeto.
*   **0.70+**: Cambio significativo de composición (usar con precaución).

---

## 👤 Créditos y Atribución
- **Autor Original:** [{meta['author']}](https://civitai.com/user/{meta['author']})
- **Cuantización Experta:** [{Config.HF_USER}](https://huggingface.co/{Config.HF_USER})

---

## 🚀 Despliegue en ComfyUI
1. **Archivo GGUF**: Mover a `ComfyUI/models/unet/`
2. **Nodos Requeridos**: Es necesario tener instalado [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).
3. **Componentes Originales**: Use el CLIP y VAE incluidos en este repo para máxima fidelidad (extraídos sin prefijos de contenedor).

---

## 📝 Nota del Autor Original
{clean_desc}

---
*GGUF Quantizer - Engineering Quality Visuals*
"""

# GGUF Quantizer Modular
class GGUFQuantizer:
    def __init__(self):
        self.api = HfApi()
        os.makedirs(Config.INPUT_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        self._validate_env()

    def _validate_env(self):
        """Valida que las herramientas necesarias estén disponibles."""
        if not os.path.exists(Config.QUANTIZE_BIN):
            print(f"⚠️ Advertencia: No se encontró el binario llama-quantize en {Config.QUANTIZE_BIN}")
            print("   Asegúrate de ejecutar setup.sh/ps1 y compilar llama.cpp.")
        
        if not os.path.exists(Config.CONVERT_SCRIPT):
            print(f"❌ Error Crítico: No se encontró el script de conversión en {Config.CONVERT_SCRIPT}")

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
        
        raw_model = os.path.join(Config.INPUT_DIR, f"{safe_name}.safetensors")
        comp_dir = os.path.join(Config.INPUT_DIR, f"components_{safe_name}")
        fp16_path = os.path.join(Config.OUTPUT_DIR, f"{safe_name}.fp16.gguf")

        try:
            # 1. Descarga
            self.download_file(meta['download_url'], raw_model)
            
            # 2. Extracción (RAM-Optimized)
            from tools.extract_components import extract_components
            extracted = extract_components(raw_model, comp_dir)
            
            # LIBERAR DISCO: Borrar modelo original tras extracción
            if os.path.exists(raw_model):
                print(f"🧹 Liberando espacio: Eliminando modelo original {os.path.basename(raw_model)}")
                os.remove(raw_model)
            
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
            # Detección inteligente de RAM para modo Low-RAM (Colab Free Tier)
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            low_ram_flag = ""
            
            # Umbral: Si hay menos de 14GB de RAM y el usuario no fuerza lo contrario
            # Colab Free tiene ~12.7 GB.
            if total_ram_gb < 14:
                print(f"⚠️ Low RAM detected ({total_ram_gb:.1f} GB). Enabling chunked processing to prevent OOM.")
                low_ram_flag = "--low-ram"
                
            print(f"📦 Converting to FP16 GGUF (Low RAM: {'ON' if low_ram_flag else 'OFF'})...")
            convert_cmd = f"{sys.executable} {Config.CONVERT_SCRIPT} --src \"{unet_path}\" --dst \"{fp16_path}\" {low_ram_flag}"
            subprocess.run(convert_cmd, shell=True, check=True)
            
            # 5. Cuantización
            for q in quants_to_process:
                print(f"⚖️ Cuantizando a {q}...")
                quant_name = f"{safe_name}.{q}.gguf"
                quant_path = os.path.join(Config.OUTPUT_DIR, quant_name)
                try:
                    subprocess.run([Config.QUANTIZE_BIN, fp16_path, quant_path, q], check=True)
                    
                    if upload_to_hf:
                        print(f"⬆️ Subiendo {quant_name}...")
                        self.api.upload_file(path_or_fileobj=quant_path, path_in_repo=quant_name, repo_id=repo_id, token=Config.HF_TOKEN)
                    else:
                        print(f"💾 Guardado localmente: {quant_path}")
                finally:
                    # Borrado inmediato del archivo cuantizado tras subida o error
                    if upload_to_hf and os.path.exists(quant_path):
                        os.remove(quant_path)

            # 6. README (Si procede)
            if upload_to_hf:
                print("📝 Actualizando documentación en HF...")
                readme = Documentation.generate_readme(meta, quants_to_process)
                self.api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id, token=Config.HF_TOKEN)

        finally:
            if upload_to_hf:
                # Limpieza garantizada de archivos pesados solo si se suben a HF
                print(f"🧹 Limpiando archivos temporales de: {meta['name']}...")
                if os.path.exists(raw_model):
                    os.remove(raw_model)
                if os.path.exists(fp16_path):
                    os.remove(fp16_path)
                if os.path.exists(comp_dir):
                    shutil.rmtree(comp_dir)
            else:
                print(f"💾 Modo local (--no-upload): Se conservan todos los archivos.")
                
            print(f"✅ Finalizado procesamiento de: {meta['name']}")
