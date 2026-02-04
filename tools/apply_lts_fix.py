import os
import sys
import re

def patch_llama_model():
    target_file = "llama.cpp/src/llama-model.cpp"
    
    if not os.path.exists(target_file):
        # Fallback for different CWD
        if os.path.exists(f"../{target_file}"):
            target_file = f"../{target_file}"
        elif os.path.exists("src/llama-model.cpp"):
            target_file = "src/llama-model.cpp"
        else:
            print(f"❌ Error: File {target_file} not found locally.")
            sys.exit(1)
        
    print(f"🔧 Applying robust fix to {target_file}...")
    
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Fix 1: load_arch throw -> warning
    # We search for the specific throw line.
    # Pattern: throw std::runtime_error("unknown model architecture...
    
    if 'LLAMA_LOG_WARN("%s: unknown architecture' in content:
         print("   ⚠️ load_arch already patched")
    else:
        # Regex replacement for robust whitespace handling
        pattern_arch = r'throw\s+std::runtime_error\s*\(\s*"unknown model architecture:.*?\);'
        replacement_warn = 'LLAMA_LOG_WARN("%s: unknown architecture \'%s\' - proceeding in tolerant mode\\n", __func__, ml.get_arch_name().c_str());'
        
        new_content = re.sub(pattern_arch, replacement_warn, content, count=1)
        
        if new_content != content:
            content = new_content
            print("   ✅ Patched load_arch (throw -> warning)")
        else:
            print("   ❌ Failed to find load_arch throw pattern via Regex")
            # Debug: print snippet where it should be
            idx = content.find("unknown model architecture")
            if idx != -1:
                print(f"   Debug context: {content[idx-50:idx+50]}")

    # Fix 2: load_hparams early return
    # We simply insert the check at the beginning of the function
    
    func_sig = "void llama_model::load_hparams(llama_model_loader & ml) {"
    
    if "skipping hparams for unknown/image architecture" in content:
        print("   ⚠️ load_hparams already patched")
    elif func_sig in content:
        replacement_code = """void llama_model::load_hparams(llama_model_loader & ml) {
    if (arch == LLM_ARCH_UNKNOWN) {
        LLAMA_LOG_WARN("%s: skipping hparams for unknown/image architecture\\n", __func__);
        return;
    }"""
        content = content.replace(func_sig, replacement_code)
        print("   ✅ Patched load_hparams (early return added)")
    else:
        print(f"   ❌ Failed to find function signature: {func_sig}")

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("✅ Robust patch process completed.")

if __name__ == "__main__":
    patch_llama_model()
