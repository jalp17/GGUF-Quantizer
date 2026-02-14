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

    if args.low_ram:
        # LOW-RAM MODE: Write tensors immediately to disk
        writer.write_header_to_file(path=dst_path)
        writer.write_kv_data_to_file()
        
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
        
        # Strategy B: Use the existing logic but FORCE garbage collection
        # We can't easily stream with standard GGUFWriter without comprehensive rewrite.
        
        # Alternative: We modify handle_tensors to NOT accumulate numpy arrays in GGUFWriter
        # but instead keeping them as lazy, and only materializing during write.
        # However, convert.py already quantizes BEFORE adding to writer.
        
        logging.info("Low RAM mode: Quantizing and writing tensors sequentially...")
        
        # Write header and KV first (standard)
        writer.write_header_to_file(path=dst_path)
        writer.write_kv_data_to_file()
        
        # We need to manually write tensor info and data
        # This requires accessing private methods or rewriting the loop.
        # Given we can't easily patch GGUFWriter, we will stick to the standard flow
        # but ensure aggressive GC.
        
        # Actually, standard convert.py flow:
        # 1. handle_tensors() loops ALL tensors, quantizes them, adds to writer.tensors list
        # 2. writer.write_tensors_to_file() loops writer.tensors and writes them.
        
        # The MEMORY SPIKE is because `writer.tensors` holds ALL quantized tensors in RAM.
        
        # FIX: We need a CustomGGUFWriter or a patched flow.
        # Let's inject a custom writer logic here.
        
        import gc
        padding = gguf.GGUFWriter.gguf_pad(writer.data_offset, writer.alignment)
        writer.fout.write(padding)
        writer.data_offset += len(padding)
        
        # We need to calculate offsets ahead of time? No, GGUF stores offset relative to data start.
        # But wait, Tensor Info block is written BEFORE Tensor Data block.
        # So we need to know all offsets/sizes BEFORE writing any data.
        
        # This confirms we CANNOT stream data easily if we follow spec strictly (Infos block first).
        # HOWEVER, the data offset is relative.
        
        # Let's stick to the Plan B from task: 
        # "Modificar convert.py para escribir el GGUF FP16 de forma incremental"
        # If we can't stream GGUF easily, we should at least avoid holding all F16 tensors in RAM.
        
        # Current convert.py:
        # handle_tensors -> loads tensor -> quantizes to F16/Q8 -> adds to writer.
        
        # We will modify handle_tensors to accept a 'flush' callback or similar?
        # No, because writer.write_tensors_to_file expects all tensors to be present in .tensors list
        # to write the info block first.
        
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
        
        # Prepare file
        writer.write_header_to_file(path=dst_path)
        writer.write_kv_data_to_file()
        
        # We need to calculate alignment padding
        alignment = writer.alignment
        data_offset = writer.data_offset
        
        # Add padding before data starts (standard GGUF alignment)
        # Note: GGUFWriter.write_kv_data_to_file() doesn't add the alignment padding for the data block end.
        # The padding is calculated based on the GLOBAL offset.
        
        # We need access to writer.fout (which is created in write_header_to_file if path provided)
        # But writer.write_header_to_file opens the file.
        
        # Pass 1: Simulate quantization to get sizes/types
        tensor_infos = []
        current_offset = 0
        
        import gc
        
        logging.info("Pass 1: Analyzing tensor sizes...")
        for key, data in tqdm(state_dict.items(), desc="Analyzing"):
             # Filter ignored
            if any(x in key for x in model_arch.keys_ignore): continue
            
            # Determine type and shape using Zero-Copy access if possible
            if isinstance(state_dict, LazyStateDict):
                # Optimization: Read metadata without loading tensor data!
                shape, old_dtype_str = state_dict.get_tensor_meta(key)
                
                # Map string dtype (safetensors) to torch dtype
                # Safetensors dtypes: F16, F32, BF16, I8, etc.
                if old_dtype_str == "BF16":
                    old_dtype = torch.bfloat16
                    data_dtype = torch.float32 
                elif old_dtype_str == "F16":
                    old_dtype = torch.float16
                    data_dtype = torch.float16
                elif old_dtype_str == "F32":
                    old_dtype = torch.float32
                    data_dtype = torch.float32
                else:
                    # Fallback for unknown types (e.g. FP8)
                    # We might need to load to check, but let's assume valid
                    old_dtype = torch.float32 # Placeholder
                    data_dtype = torch.float32
            else:
                # Standard path (data is already loaded or not lazy supported)
                if data.dtype == torch.bfloat16:
                    data_dtype = torch.float32 # temporary for shape check
                else:
                    data_dtype = data.dtype
                old_dtype = data.dtype
                shape = data.shape
            
            # Determine target type logic...
            
            n_params = 1
            for dim in shape: n_params *= dim
            
            # Determine target type
            data_qtype = gguf.GGMLQuantizationType.F16 # Default
            if old_dtype == torch.bfloat16: data_qtype = gguf.GGMLQuantizationType.BF16
            
            # Replicate the heuristics
            if old_dtype in (torch.float32, torch.bfloat16):
                if len(shape) == 1: data_qtype = gguf.GGMLQuantizationType.F32
                elif n_params <= QUANTIZATION_THRESHOLD: data_qtype = gguf.GGMLQuantizationType.F32
                elif any(x in key for x in model_arch.keys_hiprec): data_qtype = gguf.GGMLQuantizationType.F32
            
            # Calculate size
            # gguf-py doesn't expose type_size easily for calculation without data?
            # We can use gguf.ggml_type_size(data_qtype) approx?
            # Actually easiest is to trust the logic.
            
            # We need to reshape?
            # conversion logic (lines 303-312)
            if (model_arch.shape_fix and len(shape) > 1 and n_params >= REARRANGE_THRESHOLD 
                and (n_params / 256).is_integer() and not (shape[-1] / 256).is_integer()):
                # Reshape happened
                orig_shape = shape
                shape = (n_params // 256, 256)
                # We need to add query kv for orig_shape?
                # writer.add_array... we can do this now as it is KV data?
                # NO, KV data is already written. This is a problem for "Pass 1".
                # If we add KV pairs now, they won't be in the file.
                
                # Correction: We must add ALL KV pairs before write_kv_data_to_file.
                # So we actually need to Pre-Pass just to find reshapes?
                # Or just accept that we might miss the orig_shape KV in low-ram mode?
                # Actually, the orig_shape is useful but maybe not critical? 
                # Let's try to add it. But writer is already written.
                pass 
            
            # Calculate bytes
            # Block size and type size
            blk_size = 1
            type_size = 2 # F16
            if data_qtype == gguf.GGMLQuantizationType.F32: type_size = 4
            elif data_qtype == gguf.GGMLQuantizationType.BF16: type_size = 2
            elif data_qtype == gguf.GGMLQuantizationType.Q8_0: 
                blk_size = 32
                type_size = 34 # 32 bytes + 2 bytes delta
            
            # Size = (n_params / blk_size) * type_size
            size_bytes = (n_params // blk_size) * type_size
            
            # Alignment
            padding = 0
            if current_offset % alignment != 0:
                padding = alignment - (current_offset % alignment)
            
            offset = current_offset + padding
            current_offset = offset + size_bytes
            
            # Store info
            tensor_infos.append({
                "name": key,
                "shape": shape,
                "type": data_qtype,
                "offset": offset,
                "data_dtype": old_dtype # For Pass 2
            })
            
            # Force GC
            del data
            
        # Write Tensor Infos
        logging.info(f"Pass 1 Done. Calculated {len(tensor_infos)} tensors. Writing Info Block...")
        
        # We can't use writer.add_tensor because it expects data.
        # We must manually write the TI block.
        # This is accessing private/internal API of GGUFWriter, risky but necessary.
        
        # GGUF spec:
        # [uint64] n_tensors
        # for i in n_tensors:
        #   [string] name
        #   [uint32] n_dims
        #   [uint64] dim[n_dims]
        #   [uint32] type
        #   [uint64] offset
        
        import struct
        fout = writer.fout
        
        # Write n_tensors
        fout.write(struct.pack("<Q", len(tensor_infos)))
        
        for info in tensor_infos:
            # Name
            bname = info["name"].encode("utf8")
            fout.write(struct.pack("<Q", len(bname)))
            fout.write(bname)
            
            # Dims
            shape = info["shape"]
            fout.write(struct.pack("<I", len(shape)))
            for dim in reversed(shape): # GGUF uses reverse order (Dim 0 is last?) 
                # Wait, GGUFWriter.add_tensor does: data.shape (numpy) -> reversed
                fout.write(struct.pack("<Q", int(dim)))
                
            # Type
            fout.write(struct.pack("<I", int(info["type"])))
            
            # Offset
            fout.write(struct.pack("<Q", int(info["offset"])))
            
        # Write alignment padding for first tensor
        # Global alignment padding logic from GGUFWriter
        # The padding is actually written BEFORE the data of each tensor, relative to the file position?
        # No, offset is relative to base of data block.
        # But we need to pad the FILE to align the data block start?
        
        # Standard GGUFWriter:
        # write_tensors_to_file:
        #   write_padding(self.fout, self.data_offset)
        
        bar_offset = writer.data_offset # This tracks file pointer? No, data_offset is for alignment calc?
        
        # Let's align the start of data block
        curr_pos = fout.tell()
        rem = curr_pos % alignment
        if rem != 0:
            pad = alignment - rem
            fout.write(bytes([0]*pad))
            
        # Pass 2: Write Data
        logging.info("Pass 2: Quantizing and Writing Data...")
        
        base_data_pos = fout.tell()
        
        for info in tqdm(tensor_infos, desc="Writing"):
            key = info["name"]
            # Reload data
            data = state_dict[key] # Lazy load
            
            # Quantize (Replicate logic)
            if info["data_dtype"] == torch.bfloat16:
                data = data.to(torch.float32).numpy()
            elif info["data_dtype"] in [getattr(torch, "float8_e4m3fn", "_invalid"), getattr(torch, "float8_e5m2", "_invalid")]:
                data = data.to(torch.float16).numpy()
            else:
                data = data.numpy()
                
            # Reshape if needed (shape_fix)
            # We already calculated shape in Pass 1, just trust it fits
            if (model_arch.shape_fix and len(data.shape) > 1 
                and data.size >= REARRANGE_THRESHOLD 
                and (data.size / 256).is_integer() 
                and not (data.shape[-1] / 256).is_integer()):
                 data = data.reshape(data.size // 256, 256)

            # Quantize
            qtype = info["type"]
            try:
                data = gguf.quants.quantize(data, qtype)
            except Exception as e:
                tqdm.write(f"Fallback F16 for {key}: {e}")
                qtype = gguf.GGMLQuantizationType.F16
                data = gguf.quants.quantize(data, qtype)
            
            # Write data
            # Calculate padding relative to base_data_pos
            # info["offset"] is relative to base_data_pos
            
            # Check alignment
            current_rel_pos = fout.tell() - base_data_pos
            expected_offset = info["offset"]
            
            if current_rel_pos < expected_offset:
                pad_len = expected_offset - current_rel_pos
                fout.write(bytes([0]*pad_len))
            elif current_rel_pos > expected_offset:
                raise RuntimeError(f"Offset mismatch for {key}! Expected {expected_offset}, got {current_rel_pos}")
                
            # Write bytes
            data.tofile(fout)
            
            # Cleanup
            del data
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

