# Guía de Parches Modulares para `llama.cpp`

Esta carpeta contiene parches divididos por funcionalidad. Este enfoque es más robusto ante actualizaciones de `llama.cpp` que un único parche gigante.

## Automatización (Recomendado)

Se han incluido scripts para automatizar todo el proceso tanto en **Windows** como en **Linux/macOS**:

-   **Windows**: Ejecuta `./setup.ps1` en PowerShell.
-   **Linux/macOS**: Ejecuta `bash setup.sh`.

Estos scripts descargarán la versión más reciente de `llama.cpp` y aplicarán los parches en el orden correcto.

## Inventario de Parches

| Archivo | Propósito |
| :--- | :--- |
| `0001-ggml-max-name.patch` | Aumenta el límite de caracteres de los nombres de tensores (necesario para modelos de imagen). |
| `0002-llama-arch.patch` | Añade soporte para nuevas arquitecturas de imagen (Flux, SDXL, Cosmos, etc.). |
| `0003-llama-model-quant.patch` | Implementa reglas de cuantización específicas (salto de tensores sensibles). |
| `0004-llama-model-loader-tolerant.patch` | Hace que la carga de archivos GGUF sea "tolerante" a metadatos faltantes. |

## Cómo aplicarlos manualmente

Para aplicar estos parches a una versión fresca de `llama.cpp`:

1.  Navega a la raíz de tu repositorio `llama.cpp`.
2.  Ejecuta los siguientes comandos:

```bash
# Aplicar soporte de arquitecturas
git apply --verbose /ruta/a/patches/0001-llama-arch.patch

# Aplicar reglas de cuantización
git apply --verbose /ruta/a/patches/0002-llama-model.patch

# Aplicar cargador tolerante
# (Nota: Este archivo puede requerir reemplazo directo si es nuevo en tu versión)
git apply --verbose /ruta/a/patches/0003-llama-model-loader-tolerant.patch
```

## Por qué este enfoque es mejor

1.  **Aislamiento**: Si `llama.cpp` cambia drásticamente la forma en que carga modelos, solo fallará el parche `0003`, pero el soporte de arquitecturas (`0001`) seguirá funcionando.
2.  **Mantenibilidad**: Es mucho más sencillo actualizar un parche pequeño que uno de miles de líneas.
3.  **Depuración**: Si algo falla tras la cuantización, puedes revertir parches individuales para identificar al culpable.
