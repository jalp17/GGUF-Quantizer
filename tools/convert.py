# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import sys
import gguf
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import load_file, save_file

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4

class ModelTemplate:
    arch = "invalid"  # string describing architecture
    shape_fix = False # whether to reshape tensors
    keys_detect = []  # list of lists to match in state dict
    keys_banned = []  # list of keys that should mark model as invalid for conversion
    keys_hiprec = []  # list of keys that need to be kept in fp32 for some reason
    keys_ignore = []  # list of strings to ignore keys by when found

    def handle_nd_tensor(self, key, data):
        raise NotImplementedError(f"Tensor detected that exceeds dims supported by C++ code! ({key} @ {data.shape})")

class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = [
        ("transformer_blocks.0.attn.norm_added_k.weight",), # Diffusers
        ("double_blocks.0.img_attn.proj.weight",),          # Comfy/Reference
    ]
    # Ignorar siempre componentes no pertenecientes al transformer del modelo base
    keys_ignore = [
        "vae.", "first_stage_model.", 
        "conditioner.", "text_encoders.", 
        "clip_l.", "t5xxl.", # Prefijos comunes de T5/CLIP
        "guidance_in.", "img_in.", "time_in.", "vector_in." # Depende del merge, pero a menudo se ignoran si no son core
    ]
    # No banear lo que detectamos, eso era un error de lógica de la versión previa
    keys_banned = []

class ModelSD3(ModelTemplate):
    arch = "sd3"
    keys_detect = [
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.add_q_proj.weight",]

class ModelAura(ModelTemplate):
    arch = "aura"
    keys_detect = [
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",),
    ]
    keys_banned = ["joint_transformer_blocks.3.ff_context.out_projection.weight",]

class ModelHiDream(ModelTemplate):
    arch = "hidream"
    keys_detect = [
        (
            "caption_projection.0.linear.weight",
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"
        )
    ]
    keys_hiprec = [
        # nn.parameter, can't load from BF16 ver
        ".ff_i.gate.weight",
        "img_emb.emb_pos"
    ]

class CosmosPredict2(ModelTemplate):
    arch = "cosmos"
    keys_detect = [
        (
            "blocks.0.mlp.layer1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
        )
    ]
    keys_hiprec = ["pos_embedder"]
    keys_ignore = ["_extra_state", "accum_"]

class ModelHyVid(ModelTemplate):
    arch = "hyvid"
    keys_detect = [
        (
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        )
    ]

    def handle_nd_tensor(self, key, data):
        # hacky but don't have any better ideas
        path = f"./fix_5d_tensors_{self.arch}.safetensors" # TODO: somehow get a path here??
        if os.path.isfile(path):
            raise RuntimeError(f"5D tensor fix file already exists! {path}")
        fsd = {key: torch.from_numpy(data)}
        tqdm.write(f"5D key found in state dict! Manual fix required! - {key} {data.shape}")
        save_file(fsd, path)

class ModelWan(ModelHyVid):
    arch = "wan"
    keys_detect = [
        (
            "blocks.0.self_attn.norm_q.weight",
            "text_embedding.2.weight",
            "head.modulation",
        )
    ]
    keys_hiprec = [
        ".modulation" # nn.parameter, can't load from BF16 ver
    ]

class ModelLTXV(ModelTemplate):
    arch = "ltxv"
    keys_detect = [
        (
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        )
    ]
    keys_hiprec = [
        "scale_shift_table" # nn.parameter, can't load from BF16 base quant
    ]

class ModelSDXL(ModelTemplate):
    arch = "sdxl"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",
        ), # Non-diffusers
        ("label_emb.0.0.weight",),
    ]

class ModelSD1(ModelTemplate):
    arch = "sd1"
    shape_fix = True
    keys_detect = [
        ("down_blocks.0.downsamplers.0.conv.weight",),
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"
        ), # Non-diffusers
    ]

class ModelLumina2(ModelTemplate):
    arch = "lumina2"
    keys_detect = [
        ("cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight")
    ]

arch_list = [ModelFlux, ModelSD3, ModelAura, ModelHiDream, CosmosPredict2, 
             ModelLTXV, ModelHyVid, ModelWan, ModelSDXL, ModelSD1, ModelLumina2]

def is_model_arch(model, state_dict):
    # check if model is correct
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    assert not invalid, "Model architecture not allowed for conversion! (i.e. reference VS diffusers format)"
    return matched

def detect_arch(state_dict):
    model_arch = None
    for arch in arch_list:
        if is_model_arch(arch, state_dict):
            model_arch = arch()
            break
    assert model_arch is not None, "Unknown model architecture!"
    return model_arch

def parse_args():
    parser = argparse.ArgumentParser(description="Generate F16 GGUF files from single UNET")
    parser.add_argument("--src", type=str, required=True, help="Path to input .safetensors file")
    parser.add_argument("--dst", type=str, help="Path to output .gguf file")
    parser.add_argument("--low-ram", action="store_true", help="Enable low-RAM mode (slow, but works on 12GB envs)")
    parser.add_argument("--extract-t5", action="store_true", help="Extract T5 encoder to a separate safetensors file (for Flux/SD3)")
    args = parser.parse_args()

    if not os.path.isfile(args.src):
        parser.error("No input provided!")

    return args

def strip_prefix(state_dict):
    # prefix for mixed state dict
    prefix = None
    for pfx in ["model.diffusion_model.", "model."]:
        if any([x.startswith(pfx) for x in state_dict.keys()]):
            prefix = pfx
            break

    # prefix for uniform state dict
    if prefix is None:
        for pfx in ["net."]:
            if all([x.startswith(pfx) for x in state_dict.keys()]):
                prefix = pfx
                break

    # strip prefix if found
    if prefix is not None:
        logging.info(f"State dict prefix found: '{prefix}'")
        if isinstance(state_dict, LazyStateDict):
            # Optimización Zero-Copy: Devolver nueva vista Lazy con prefijo
            return LazyStateDict(state_dict.path, prefix=prefix)
        else:
            sd = {}
            for k, v in state_dict.items():
                if prefix not in k:
                    continue
                k = k.replace(prefix, "")
                sd[k] = v
            return sd
    else:
        logging.debug("State dict has no prefix")
        return state_dict

class LazyStateDict:
    def __init__(self, path, prefix=""):
        self.path = path
        self.f = safe_open(path, framework="pt", device="cpu")
        self.prefix = prefix
        
        # Filtrar claves si hay prefijo (forzar lista para evitar iteradores)
        all_keys = list(self.f.keys())
        if prefix:
            self.keys_list = [k[len(prefix):] for k in all_keys if k.startswith(prefix)]
        else:
            self.keys_list = all_keys
            
    def keys(self): return self.keys_list
    def items(self):
        for k in self.keys_list:
            yield k, self.get_tensor(k)

    def values(self):
        for k in self.keys_list:
            yield self.get_tensor(k)

    def values(self):
        for k in self.keys_list:
            yield self.get_tensor(k)
            
    def __getitem__(self, key): return self.get_tensor(key)
    def __contains__(self, key): return key in self.keys_list
    
    def get_tensor(self, key):
        # Reconstruct original key
        orig_key = self.prefix + key
        tensor = self.f.get_tensor(orig_key)
        return tensor

    def get_tensor_meta(self, key):
        """Devuelve (shape, dtype_str) sin cargar los datos del tensor."""
        orig_key = self.prefix + key
        slice_obj = self.f.get_slice(orig_key)
        return slice_obj.get_shape(), slice_obj.get_dtype()

    def metadata(self):
        """Retorna el diccionario de metadatos del archivo safetensors."""
        return self.f.metadata()

def load_state_dict(path):
    if any(path.endswith(x) for x in [".ckpt", ".pt", ".bin", ".pth"]):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        for subkey in ["model", "module"]:
            if subkey in state_dict:
                state_dict = state_dict[subkey]
                break
        if len(state_dict) < 20:
            raise RuntimeError(f"pt subkey load failed: {state_dict.keys()}")
    else:
        # Usar carga perezosa para safetensors
        state_dict = LazyStateDict(path)

    return strip_prefix(state_dict)

    return strip_prefix(state_dict)

def handle_tensors(writer, state_dict, model_arch):
    name_lengths = tuple(sorted(
        ((key, len(key)) for key in state_dict.keys()),
        key=lambda item: item[1],
        reverse=True,
    ))
    if not name_lengths:
        return
    max_name_len = name_lengths[0][1]
    if max_name_len > MAX_TENSOR_NAME_LENGTH:
        bad_list = ", ".join(f"{key!r} ({namelen})" for key, namelen in name_lengths if namelen > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(f"Can only handle tensor names up to {MAX_TENSOR_NAME_LENGTH} characters. Tensors exceeding the limit: {bad_list}")
    for key, data in tqdm(state_dict.items()):
        old_dtype = data.dtype

        if any(x in key for x in model_arch.keys_ignore):
            tqdm.write(f"Filtering ignored key: '{key}'")
            continue

        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
        # this is so we don't break torch 2.0.X
        elif data.dtype in [getattr(torch, "float8_e4m3fn", "_invalid"), getattr(torch, "float8_e5m2", "_invalid")]:
            data = data.to(torch.float16).numpy()
        else:
            data = data.numpy()

        n_dims = len(data.shape)
        data_shape = data.shape
        if old_dtype == torch.bfloat16:
            data_qtype = gguf.GGMLQuantizationType.BF16
        # elif old_dtype == torch.float32:
        #     data_qtype = gguf.GGMLQuantizationType.F32
        else:
            data_qtype = gguf.GGMLQuantizationType.F16

        # The max no. of dimensions that can be handled by the quantization code is 4
        if len(data.shape) > MAX_TENSOR_DIMS:
            model_arch.handle_nd_tensor(key, data)
            continue # needs to be added back later

        # get number of parameters (AKA elements) in this tensor
        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size

        if old_dtype in (torch.float32, torch.bfloat16):
            if n_dims == 1:
                # one-dimensional tensors should be kept in F32
                # also speeds up inference due to not dequantizing
                data_qtype = gguf.GGMLQuantizationType.F32

            elif n_params <= QUANTIZATION_THRESHOLD:
                # very small tensors
                data_qtype = gguf.GGMLQuantizationType.F32

            elif any(x in key for x in model_arch.keys_hiprec):
                # tensors that require max precision
                data_qtype = gguf.GGMLQuantizationType.F32

        if (model_arch.shape_fix                        # NEVER reshape for models such as flux
            and n_dims > 1                              # Skip one-dimensional tensors
            and n_params >= REARRANGE_THRESHOLD         # Only rearrange tensors meeting the size requirement
            and (n_params / 256).is_integer()           # Rearranging only makes sense if total elements is divisible by 256
            and not (data.shape[-1] / 256).is_integer() # Only need to rearrange if the last dimension is not divisible by 256
        ):
            orig_shape = data.shape
            data = data.reshape(n_params // 256, 256)
            writer.add_array(f"comfy.gguf.orig_shape.{key}", tuple(int(dim) for dim in orig_shape))

        try:
            data = gguf.quants.quantize(data, data_qtype)
        except (AttributeError, gguf.QuantError) as e:
            tqdm.write(f"falling back to F16: {e}")
            data_qtype = gguf.GGMLQuantizationType.F16
            data = gguf.quants.quantize(data, data_qtype)

        new_name = key # do we need to rename?

        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        tqdm.write(f"{f'%-{max_name_len + 4}s' % f'{new_name}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

        writer.add_tensor(new_name, data, raw_dtype=data_qtype)

def convert_file(path, dst_path=None, interact=True, overwrite=False):
    # load & run model detection logic
    state_dict = load_state_dict(path)
    model_arch = detect_arch(state_dict)
    logging.info(f"* Architecture detected from input: {model_arch.arch}")

    # detect & set dtype for output file
    if isinstance(state_dict, LazyStateDict):
        # Optimización: No cargar tensores, solo mirar metadatos
        dtypes = []
        for k in state_dict.keys():
            _, dt = state_dict.get_tensor_meta(k)
            # dt es un string de safetensors, convertir a torch dtype si es necesario para compatibilidad
            # pero aquí solo necesitamos saber si es BF16 o F16.
            # safe_open dtypes: 'F32', 'F16', 'BF16'
            if dt == 'BF16': dtypes.append(torch.bfloat16)
            elif dt == 'F16': dtypes.append(torch.float16)
            else: dtypes.append(torch.float32)
    else:
        dtypes = [x.dtype for x in state_dict.values()]
        
    dtypes_count = {x:dtypes.count(x) for x in set(dtypes)}
    main_dtype = max(dtypes_count, key=dtypes_count.get)

    if main_dtype == torch.bfloat16:
        ftype_name = "BF16"
        ftype_gguf = getattr(gguf, "LlamaFileType", getattr(gguf, "FileType", None))
        if ftype_gguf: ftype_gguf = getattr(ftype_gguf, "MOSTLY_BF16", 12) # 12 = BF16 in GGUF
    else:
        ftype_name = "F16"
        ftype_gguf = getattr(gguf, "LlamaFileType", getattr(gguf, "FileType", None))
        if ftype_gguf: ftype_gguf = getattr(ftype_gguf, "MOSTLY_F16", 1)  # 1 = F16 in GGUF

    if dst_path is None:
        dst_path = f"{os.path.splitext(path)[0]}-{ftype_name}.gguf"
    elif "{ftype}" in dst_path: # lcpp logic
        dst_path = dst_path.replace("{ftype}", ftype_name)

    if os.path.isfile(dst_path) and not overwrite:
        if interact:
            input("Output exists enter to continue or ctrl+c to abort!")
        else:
            raise OSError("Output exists and overwriting is disabled!")

    # handle actual file
    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    
    # Versión de cuantización (segura)
    q_version = getattr(gguf, "GGML_QUANT_VERSION", 2)
    writer.add_quantization_version(q_version)
    
    if ftype_gguf is not None:
        writer.add_file_type(ftype_gguf)

    # [NEW] Copiar metadatos del Safetensors a GGUF (Modo Compatible)
    if hasattr(state_dict, 'metadata'):
        meta = state_dict.metadata()
        if meta:
            logging.info(f"Procesando {len(meta)} metadatos de safetensors...")
            for k, v in meta.items():
                # Filtrar metadatos técnicos internos de safetensors o redundantes
                if k.startswith("modelspec.") or k in ["format", "ss_tag_frequency", "ss_tags_card"]: continue
                
                # Mapeo de llaves comunes a estándar GGUF para mejor compatibilidad
                # stable-diffusion.cpp y otros buscan llaves específicas en la raíz
                key = k
                if k == "ss_output_name": key = "general.name"
                elif k == "ss_sd_model_name": key = "general.base_model.name"
                
                # Inyectar valor (limitado a strings por simplicidad en metadatos ST)
                try:
                    writer.add_string(key, str(v))
                except Exception as e:
                    logging.warning(f"No se pudo añadir metadato {key}: {e}")

    # Inyectar metadatos técnicos REQUERIDOS por stable-diffusion.cpp y llama.cpp
    # El cargador necesita saber context_length, embedding_length y block_count para reservar memoria
    if model_arch.arch in ["sdxl", "sd1", "sd3", "flux"]:
        logging.info(f"Configurando metadatos de arquitectura para: {model_arch.arch}")
        
        # Prefijo estándar: {arch}.{key}
        arch_prefix = model_arch.arch
        
        # Parámetros por arquitectura (Ajustados según especificación de SD)
        if model_arch.arch == "sd1":
            writer.add_uint32(f"{arch_prefix}.context_length", 77)
            writer.add_uint32(f"{arch_prefix}.embedding_length", 768)
            writer.add_uint32(f"{arch_prefix}.block_count", 12)
        elif model_arch.arch == "sdxl":
            writer.add_uint32(f"{arch_prefix}.context_length", 77)
            writer.add_uint32(f"{arch_prefix}.embedding_length", 2048) # SDXL usa CLIP-L + OpenCLIP-G
            writer.add_uint32(f"{arch_prefix}.block_count", 20)
        elif model_arch.arch == "flux":
            # Flux requiere parámetros específicos para el flujo de atención
            writer.add_uint32(f"{arch_prefix}.context_length", 256)
            writer.add_uint32(f"{arch_prefix}.embedding_length", 4096) # T5 XXL
            writer.add_uint32(f"{arch_prefix}.block_count", 19) # 19 double blocks
        elif model_arch.arch == "sd3":
            writer.add_uint32(f"{arch_prefix}.context_length", 256)
            writer.add_uint32(f"{arch_prefix}.embedding_length", 1536)
            writer.add_uint32(f"{arch_prefix}.block_count", 24)

        # Metadato de arquitectura general (CRÍTICO para cargadores)
        writer.add_string("general.architecture", model_arch.arch)
        writer.add_string("general.name", os.path.basename(dst_path))
        
    else:
        # Modo normal (RAM alta): Todo se mantiene en memoria hasta save_gguf() o similar
        pass
        
        # Custom implementation of write_tensors_to_file for low RAM
        # We perform quantization and write TENSOR INFO + TENSOR DATA individually
        
        # 1. First pass: Calculate offsets and write tensor info (metadata)
        # In standardized GGUF, tensor info block comes before data.
        # But we don't know the exact size of quantized data without quantizing!
        # This is tricky. GGUF spec requires: Header -> KV Data -> Tensor Infos -> Tensor Data
        
        # Standard GGUFWriter accumulates all data to calculate offsets.
        # To do this in low-RAM, we must quantize TWICE or estimate size perfectly.
        # Fortunately, gguf library's quantize() returns predictable sizes for most types.
        
        # BUT wait: GGUFWriter in gguf.py is not designed for streaming. 
        # It stores self.tensors = [].
        
        # Alternative: We modify handle_tensors to NOT accumulate numpy arrays in GGUFWriter
        # but instead keeping them as lazy, and only materializing during write.
        # However, convert.py already quantizes BEFORE adding to writer.
        
        # Actually, standard convert.py flow:
        # 1. handle_tensors() loops ALL tensors, quantizes them, adds to writer.tensors list
        # 2. writer.write_tensors_to_file() loops writer.tensors and writes them.
        
        # The MEMORY SPIKE is because `writer.tensors` holds ALL quantized tensors in RAM.
        
        # Solution: Two-pass approach with minimal RAM
        # Pass 1: Compute sizes and offsets (without storing data) -> Write Tensor Infos
        # Pass 2: Quantize and Write Data
        
        # This requires hacking GGUFWriter. 
        # Let's look at how handle_tensors is called.
        pass

    # [NEW] Flux T5 Extraction
    if args.extract_t5:
        t5_keys = [k for k in state_dict.keys() if "t5xxl" in k.lower() or "t5." in k.lower()]
        if t5_keys:
            t5_path = dst_path.replace(".gguf", ".t5xxl.safetensors")
            logging.info(f"✨ Extracting {len(t5_keys)} T5 tensors to {t5_path}...")
            t5_sd = {}
            # Si es Lazy, cargamos solo lo necesario
            for k in tqdm(t5_keys, desc="Extracting T5"):
                t5_sd[k] = state_dict[k]
            
            # Guardar usando safetensors (si ya está cargado, state_dict[k] es tensor)
            # Nota: save_file de safetensors maneja dicts de tensores
            save_file(t5_sd, t5_path)
            logging.info("✅ T5 extraction complete.")
        else:
            logging.warning("⚠️ --extract-t5 requested but no T5 keys found in source.")

    if not args.low_ram:
        handle_tensors(writer, state_dict, model_arch)
        writer.write_header_to_file(path=dst_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
    else:
        # LOW RAM IMPLEMENTATION
        # 1. Calculate layout without data
        # 2. Write Headers + Infos
        # 3. Write Data
        
        logging.info("🚀 Low-RAM Mode enabled: Two-pass processing (Analyze -> Write)")
        
        # Pass 1: Analyze tensor sizes and add to writer info
        logging.info("Pass 1: Analyzing tensor sizes...")
        tensor_keys = []
        for key in tqdm(state_dict.keys(), desc="Analyzing"):
            if any(x in key for x in model_arch.keys_ignore): continue
            
            # Use Zero-Copy metadata if available
            if isinstance(state_dict, LazyStateDict):
                res = state_dict.get_tensor_meta(key)
                if res is None:
                    logging.warning(f"No meta for {key}")
                    continue
                shape, old_dtype_str = res
                if old_dtype_str == "BF16": old_dtype = torch.bfloat16
                elif old_dtype_str == "F16": old_dtype = torch.float16
                elif old_dtype_str == "F32": old_dtype = torch.float32
                else: old_dtype = torch.float32 
            else:
                raw_tensor = state_dict[key]
                old_dtype = raw_tensor.dtype
                shape = raw_tensor.shape
                del raw_tensor
            
            n_params = 1
            for dim in shape: n_params *= dim
            
            # Determine target type
            data_qtype = gguf.GGMLQuantizationType.F16 
            if old_dtype == torch.bfloat16: data_qtype = gguf.GGMLQuantizationType.BF16
            
            if old_dtype in (torch.float32, torch.bfloat16):
                if len(shape) == 1 or n_params <= QUANTIZATION_THRESHOLD or any(x in key for x in model_arch.keys_hiprec):
                    data_qtype = gguf.GGMLQuantizationType.F32
            
            # Handle shape fix (Reshape logic)
            if (model_arch.shape_fix and len(shape) > 1 and n_params >= REARRANGE_THRESHOLD 
                and (n_params / 256).is_integer() and not (shape[-1] / 256).is_integer()):
                # Preserve original shape in KV metadata
                writer.add_array(f"{key}.orig_shape", list(shape))
                shape = (n_params // 256, 256)
            
            # Calculate bytes
            blk_size = 1
            type_size = 2 # F16/BF16
            if data_qtype == gguf.GGMLQuantizationType.F32: type_size = 4
            
            size_bytes = (n_params // blk_size) * type_size
            
            # Register in writer (Info only, no data yet)
            # We map torch dtype to a dummy numpy dtype for the API
            np_dtype = np.float16
            if data_qtype == gguf.GGMLQuantizationType.F32: np_dtype = np.float32
            
            writer.add_tensor_info(key, shape, np_dtype, size_bytes, raw_dtype=data_qtype)
            tensor_keys.append((key, old_dtype, data_qtype))

        # Write Headers + KV + Tensor Info block
        logging.info(f"Pass 1 Done. Writing GGUF Headers and Metadata to {dst_path}...")
        writer.write_header_to_file(path=dst_path)
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        
        # Prepare for raw data writing
        fout = writer.fout[0]
        alignment = writer.data_alignment
        
        # Align data block start
        curr_pos = fout.tell()
        rem = curr_pos % alignment
        if rem != 0:
            fout.write(bytes([0] * (alignment - rem)))
        
        base_data_pos = fout.tell()
        
        # Pass 2: Quantize and Write Data
        logging.info("Pass 2: Quantizing and Writing Data sequentially...")
        import gc
        for i, (key, old_dtype, target_qtype) in enumerate(tqdm(tensor_keys, desc="Writing")):
            data = state_dict[key] # Lazy reload
            
            # Pre-conversion
            if old_dtype == torch.bfloat16:
                data = data.to(torch.float32).numpy()
            elif old_dtype in [getattr(torch, "float8_e4m3fn", "_invalid"), getattr(torch, "float8_e5m2", "_invalid")]:
                data = data.to(torch.float16).numpy()
            else:
                data = data.numpy()
                
            # Reshape if needed
            if (model_arch.shape_fix and len(data.shape) > 1 
                and data.size >= REARRANGE_THRESHOLD 
                and (data.size / 256).is_integer() 
                and not (data.shape[-1] / 256).is_integer()):
                 data = data.reshape(data.size // 256, 256)

            # Quantize
            data = gguf.quants.quantize(data, target_qtype)
            
            # Standard GGUF alignment: each tensor data must be aligned
            # GGUFWriter info block stores OFFSET relative to data block start.
            # We must ensure our write position matches the calculated offset.
            # But wait, we didn't store the calculated offset because GGUFWriter calculated it during write_ti!
            
            # Actually, GGUFWriter's write_ti_data_to_file calculates offsets linearly with padding.
            # We must replicate the same padding here.
            curr_rel_pos = fout.tell() - base_data_pos
            if curr_rel_pos % alignment != 0:
                fout.write(bytes([0] * (alignment - (curr_rel_pos % alignment))))
            
            # Write to file
            data.tofile(fout)
            
            del data
            if i % 10 == 0:
                gc.collect()

    writer.close()

    fix = f"./fix_5d_tensors_{model_arch.arch}.safetensors"
    if os.path.isfile(fix):
        logging.warning(f"\n### Warning! Fix file found at '{fix}'")
        logging.warning(" you most likely need to run 'fix_5d_tensors.py' after quantization.")

    return dst_path, model_arch

if __name__ == "__main__":
    args = parse_args()
    convert_file(args.src, args.dst)

