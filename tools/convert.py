# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import gguf
import torch
import logging
import argparse
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
        ("transformer_blocks.0.attn.norm_added_k.weight",),
        ("double_blocks.0.img_attn.proj.weight",),
    ]
    keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight",]

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
        
        # Filtrar claves si hay prefijo
        all_keys = self.f.keys()
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
    dtypes = [x.dtype for x in state_dict.values()]
    dtypes = {x:dtypes.count(x) for x in set(dtypes)}
    main_dtype = max(dtypes, key=dtypes.get)

    if main_dtype == torch.bfloat16:
        ftype_name = "BF16"
        ftype_gguf = gguf.LlamaFileType.MOSTLY_BF16
    # elif main_dtype == torch.float32:
    #     ftype_name = "F32"
    #     ftype_gguf = None
    else:
        ftype_name = "F16"
        ftype_gguf = gguf.LlamaFileType.MOSTLY_F16

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
    
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    if ftype_gguf is not None:
        writer.add_file_type(ftype_gguf)

    # Inyectar metadatos obligatorios para arquitecturas de imagen (evita SIGABRT en llama.cpp)
    if model_arch.arch in ["sdxl", "sd1", "sd3", "flux"]:
        # Valores estándar para que el cargador de llama.cpp no explote
        writer.add_uint32(f"{model_arch.arch}.context_length", 77)
        writer.add_uint32(f"{model_arch.arch}.embedding_length", 768 if model_arch.arch == "sd1" else 2048)
        writer.add_uint32(f"{model_arch.arch}.block_count", 12 if model_arch.arch == "sd1" else 20)
        logging.info(f"Inyectando metadatos para {model_arch.arch}...")
        
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
        for key, data in tqdm(state_dict.items(), desc="Analyzing"):
            if any(x in key for x in model_arch.keys_ignore): continue
            
            # Use Zero-Copy metadata if available
            if isinstance(state_dict, LazyStateDict):
                shape, old_dtype_str = state_dict.get_tensor_meta(key)
                if old_dtype_str == "BF16": old_dtype = torch.bfloat16
                elif old_dtype_str == "F16": old_dtype = torch.float16
                elif old_dtype_str == "F32": old_dtype = torch.float32
                else: old_dtype = torch.float32 
            else:
                old_dtype = data.dtype
                shape = data.shape
            
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
            
            del data # If it was loaded

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
        for key, old_dtype, target_qtype in tqdm(tensor_keys, desc="Writing"):
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
            if tensor_keys.index((key, old_dtype, target_qtype)) % 10 == 0:
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

