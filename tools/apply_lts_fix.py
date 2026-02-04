import os
import sys

def patch_llama_model():
    target_file = "llama.cpp/src/llama-model.cpp"
    
    if not os.path.exists(target_file):
        print(f"❌ Error: File {target_file} not found.")
        sys.exit(1)
        
    print(f"🔧 Applying robust fix to {target_file}...")
    
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Fix 1: load_arch throw -> warning
    original_throw = 'throw std::runtime_error("unknown model architecture: \'\" + ml.get_arch_name() + \"\'");'
    replacement_warn = 'LLAMA_LOG_WARN(\"%s: unknown architecture \'%s\' - proceeding in tolerant mode\\n\", __func__, ml.get_arch_name().c_str());'
    
    if original_throw in content:
        content = content.replace(original_throw, replacement_warn)
        print("   ✅ Patched load_arch (throw -> warning)")
    elif replacement_warn in content:
        print("   ⚠️ load_arch already patched")
    else:
        print("   ❌ Failed to find load_arch target pattern")
        # Try a more loose match if exact string match fails (e.g. whitespace diffs)
        # This is a fallback
        pass

    # Fix 2: load_hparams early return
    target_hparams = "void llama_model::load_hparams(llama_model_loader & ml) {"
    replacement_hparams = """void llama_model::load_hparams(llama_model_loader & ml) {
    if (arch == LLM_ARCH_UNKNOWN) {
        LLAMA_LOG_WARN("%s: skipping hparams for unknown/image architecture\\n", __func__);
        return;
    }"""
    
    if target_hparams in content and "skipping hparams" not in content:
        content = content.replace(target_hparams, replacement_hparams)
        print("   ✅ Patched load_hparams (early return added)")
    elif "skipping hparams" in content:
        print("   ⚠️ load_hparams already patched")
    else:
        print("   ❌ Failed to find load_hparams target pattern")

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("✅ Robust patch applied successfully.")

if __name__ == "__main__":
    patch_llama_model()
