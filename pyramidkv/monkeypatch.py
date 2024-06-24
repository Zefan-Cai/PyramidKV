import torch
from typing import Optional, Tuple, Dict, Any

from importlib.metadata import version
import warnings
import transformers

from pyramidkv.llama_model import llama_flash_attn2_forward_PyramidKV,llama_flash_attn2_forward_H2O,llama_flash_attn2_forward_SnapKV,llama_flash_attn2_forward_StreamingLLM
from pyramidkv.llama_model import llama_attn_forward_PyramidKV,llama_attn_forward_H2O,llama_attn_forward_SnapKV,llama_attn_forward_StreamingLLM
from pyramidkv.llama_model import llama_sdpa_attn_forward_PyramidKV,llama_sdpa_attn_forward_H2O,llama_sdpa_attn_forward_SnapKV,llama_sdpa_attn_forward_StreamingLLM

from pyramidkv.mistral_model import mistral_flash_attn2_forward_PyramidKV,mistral_flash_attn2_forward_H2O,mistral_flash_attn2_forward_SnapKV,mistral_flash_attn2_forward_StreamingLLM
from pyramidkv.mistral_model import mistral_attn_forward_PyramidKV,mistral_attn_forward_H2O,mistral_attn_forward_SnapKV,mistral_attn_forward_StreamingLLM
from pyramidkv.mistral_model import mistral_sdpa_attn_forward_PyramidKV,mistral_sdpa_attn_forward_H2O,mistral_sdpa_attn_forward_SnapKV,mistral_sdpa_attn_forward_StreamingLLM

from pyramidkv.llama_model import prepare_inputs_for_generation_llama

from pyramidkv.mistral_model import prepare_inputs_for_generation_mistral


def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def cache_update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

    Parameters:
        key_states (`torch.Tensor`):
            The new key states to cache.
        value_states (`torch.Tensor`):
            The new value states to cache.
        layer_idx (`int`):
            The index of the layer to cache the states for.
        cache_kwargs (`Dict[str, Any]`, `optional`):
            Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

    Return:
        A tuple containing the updated key and value states.
    """
    # Update the number of seen tokens
    if layer_idx == 0:
        self._seen_tokens += key_states.shape[-2]

    # Update the cache
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
    else:
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states.to(self.key_cache[layer_idx].device)], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states.to(self.value_cache[layer_idx].device)], dim=-2)

    return self.key_cache[layer_idx], self.value_cache[layer_idx]


def replace_cache():
    transformers.DynamicCache.update = cache_update

def replace_llama(method):
    transformers_version = check_version()
    version_list = ['4.41']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with PyramidKV. PyramidKV is tested with Transformers version {version_list}.")
    
    
    # Pyramid KV method
   
    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_PyramidKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_PyramidKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_PyramidKV

    elif method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_StreamingLLM
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_StreamingLLM
        
    elif method == "h2o":
        print("Using H2O!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_H2O
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_H2O
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_H2O
        
    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_SnapKV
        
        
    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama


    


def replace_mistral(method):
    transformers_version = check_version()
    version_list = ['4.41']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with PyramidKV. PyramidKV is tested with Transformers version {version_list}.")
    
    
    # Pyramid KV method
    
    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_PyramidKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_PyramidKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_PyramidKV
    
    elif method == "streamingllm":
        print("Using StreamingLLM!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_StreamingLLM
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_StreamingLLM
        
    elif method == "h2o":
        print("Using H2O!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_H2O
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_H2O
        
    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = mistral_attn_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV
        transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = mistral_sdpa_attn_forward_SnapKV
        
        
    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral
