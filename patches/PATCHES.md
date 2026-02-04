# Guía de Parches Modulares para `llama.cpp`

Esta carpeta contiene parches divididos por funcionalidad. Este enfoque es practico ante actualizaciones de `llama.cpp`.

## Automatización (Recomendado)

Se han incluido scripts para automatizar todo el proceso tanto en **Windows** como en **Linux/macOS**:

-   **Windows**: Ejecuta `./setup.ps1` en PowerShell.
-   **Linux/macOS**: Ejecuta `bash setup.sh`.

Estos scripts descargarán la versión más reciente de `llama.cpp` y aplicarán los parches en el orden correcto.

## Inventario de Parches

| Archivo | Propósito |
| :--- | :--- |
| `0001-ggml-max-name.patch` | Aumenta el límite de caracteres de los nombres de tensores (necesario para modelos de imagen). |
| `0002-llama-arch.patch` | Añade soporte para arquitecturas (Flux, SDXL, Cosmos, etc.) y hace que el mapeo de tensores sea tolerante. |
| `0003-llama-model-quant.patch` | Implementa reglas de cuantización específicas y bypass para arquitecturas desconocidas. |
| `0004-llama-model-loader-tolerant.patch` | Hace que la carga de archivos GGUF sea tolerante a metadatos faltantes. |

## Cómo aplicarlos manualmente

Para aplicar estos parches a una versión de `llama.cpp`:

1.  Navega a la raíz de tu repositorio `llama.cpp`.
2.  Ejecuta los siguientes comandos:

```bash
# 1. Aumentar límite de nombres de tensores
git apply --verbose patches/0001-ggml-max-name.patch

# 2. Soporte de arquitecturas de imagen
git apply --verbose patches/0002-llama-arch.patch

# 3. Reglas de cuantización inteligente
git apply --verbose patches/0003-llama-model-quant.patch

# 4. Cargador tolerante (para modelos no-LLM)
git apply --verbose patches/0004-llama-model-loader-tolerant.patch
```
