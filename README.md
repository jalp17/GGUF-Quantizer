# GGUF Quantizer: Ecosistema Modular de Cuantización

Este proyecto es un conjunto de herramientas para la extracción, conversión y cuantización de modelos de imagen (SDXL, Illustrious, Pony, Flux) al formato **GGUF**. Está diseñado para ser modular, local-first y compatible con el ecosistema de **ComfyUI-GGUF**.

## Características Principales

- **"Unrestricted" Quantization**: Soporte para cuantizar modelos de imagen sin las restricciones de metadatos estándar de LLM.
- **Multiplataforma**: Funciona en **Windows** y **Linux/macOS**.
- **Flujo Basado en Parches**: En lugar de modificar el código de `llama.cpp` a mano, utilizamos parches modulares y scripts de automatización para mantener la compatibilidad con las últimas versiones oficiales.

## Arquitectura del Proyecto

- **`patches/`**: Parches modulares para `llama.cpp`. Corrigen límites de nombres de tensores, añaden arquitecturas y permiten la carga tolerante de metadatos.
- **`tools/`**: Scripts de utilidad para diagnóstico, pre-procesamiento y conversión.
- **`quantizer/`**: El núcleo de la automatización (`quantizer.py`). Gestiona el flujo desde descarga hasta subida.
- **`setup.sh` / `setup.ps1`**: Scripts de configuración automática para preparar el entorno de cuantización.

## Instalación y Preparación

### 1. Requisitos de Python
```bash
pip install -r requirements.txt
```

### 2. Configuración de Secretos
Crea un archivo `.env` en la raíz del proyecto con tus credenciales:
```env
HF_TOKEN=tu_token_de_huggingface
HF_USER=tu_usuario_de_huggingface
CIVITAI_API_KEY=tu_api_key_de_civitai
```

## Preparación de llama.cpp (Automático)

Para que el proceso funcione, debes compilar una versión de `llama.cpp` con nuestros parches de imagen.

### En Windows (PowerShell):
```powershell
./setup.ps1
```

### En Linux/macOS (Bash):
```bash
bash setup.sh
```

Estos scripts descargarán el repositorio oficial, aplicarán los parches de la carpeta `patches/` y te dejarán listo para compilar con `cmake`.

## Uso Local

El script `process.py` es el punto de entrada principal.

### Procesar URLs directas:
```bash
python process.py https://civitai.com/models/ID_DEL_MODELO
```

### Procesar desde un archivo:
```bash
python process.py -f links.txt
```

### Opciones avanzadas:
- **Solo Local (sin subir a HF)**: `python process.py URL --no-upload`
- **Cuantización personalizada**: `python process.py URL -q Q4_K_M,Q8_0`

## Créditos

Este proyecto está basado en:
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)**: El motor de cuantización.
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)**: La implementación de referencia.
- Desarrollo asistido por **Antigravity (Advanced Agentic Coding)**.

---
*GGUF Quantizer - Comprimiendo la inteligencia artificial, píxel a píxel.*
