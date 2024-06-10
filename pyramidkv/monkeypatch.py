from importlib.metadata import version
import warnings
import transformers

from pyramidkv.llama_model import llama_flash_attn2_forward_pyramid

from pyramidkv.llama_model import llama_model_forward

from pyramidkv.llama_model import prepare_inputs_for_generation_llama



def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def replace_llama(method):
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with PyramidKV. PyramidKV is tested with Transformers version {version_list}.")
    
    
    # Pyramid KV method
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_pyramid
    
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama
   
    transformers.models.llama.modeling_llama.LlamaModel.forward= llama_model_forward
    


    
