# 🏭 GGUF Quantizer: Ecosistema Modular de Cuantización

Este proyecto es un conjunto de herramientas para la extracción, conversión y cuantización de modelos de imagen (SDXL, Illustrious, Pony, Flux) al formato **GGUF**. Está diseñado para ser modular, local-first y compatible con el ecosistema de **ComfyUI-GGUF**.

## 🏗️ Arquitectura del Proyecto

El repositorio está organizado de forma modular para facilitar el mantenimiento y la portabilidad:

- **`core/`**: Contiene los archivos fuente modificados de `llama.cpp` (`llama-model.cpp` y `llama-arch.cpp`). Estos archivos incluyen los parches de tolerancia de tensores y mapeo de arquitectura necesarios para modelos de imagen.
- **`tools/`**: Scripts de utilidad para diagnóstico y pre-procesamiento:
    - `extract_components.py`: Separa UNET, CLIP y VAE de un archivo `.safetensors`.
    - `model_scanner.py`: Analiza la estructura de tensores de cualquier modelo.
    - `inspect_tensors.py`: Compara modelos originales con extraídos para asegurar integridad.
- **`quantizer/`**: El núcleo de la automatización (`quantizer.py`). Gestiona el flujo completo: descarga -> extracción -> conversión FP16 -> cuantización K-Quants -> subida a Hugging Face.
- **`core/`**: Parches de `llama.cpp`.
- **`process.py`**: El punto de entrada principal para ejecuciones en máquina local.

## 🚀 Instalación y Preparación

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

## 🛠️ Compilación de llama.cpp (Versión Parcheada)

Para que el proceso funcione, debes compilar `llama.cpp` utilizando los archivos fuente de la carpeta `core/`.

1. Descarga `llama.cpp` original en una subcarpeta: `git clone https://github.com/ggerganov/llama.cpp`.
2. Sustituye `llama.cpp/src/llama-model.cpp` y `llama.cpp/src/llama-arch.cpp` por los archivos en nuestra carpeta `core/`.
3. Compila (Ejemplo Windows con MSYS2/CMake):
```bash
cd llama.cpp
mkdir build && cd build
cmake .. -G "Ninja" -DGGML_CUDA=ON  # O la opción que prefieras
cmake --build . --config Release --target llama-quantize
```

## 📋 Uso Local

El script `process.py` es el punto de entrada principal. Soporta URLs directas o archivos de texto.

### Procesar URLs directas:
```bash
python process.py https://civitai.com/models/ID_DEL_MODELO
```

### Procesar desde un archivo:
Crea un archivo (ej: `links.txt`) con una URL por línea y ejecuta:
```bash
python process.py -f links.txt
```

### Opciones avanzadas:
- **Solo Local (sin subir a HF)**: `python process.py URL --no-upload`
- **Cuantización personalizada**: `python process.py URL -q Q4_K_M,Q8_0`
- **Combinado**: `python process.py -f links.txt -q Q4_K_M --no-upload`

El script se encargará de:
1. Descargar el modelo usando tu API Key.
2. Extraer los componentes (CLIP, VAE).
3. Convertir el UNET a GGUF FP16.
4. Generar las cuantizaciones seleccionadas.
5. Subir todo a un nuevo repositorio en tu cuenta de Hugging Face con un **README.md** (si no se usa `--no-upload`).

## 🤝 Créditos
Este proyecto esta basado en:
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)**: El motor de cuantización.
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)**: La implementación de referencia para GGUF en Stable Diffusion.
---
*GGUF Quantizer - Comprimiendo la inteligencia artificial, píxel a píxel.*
