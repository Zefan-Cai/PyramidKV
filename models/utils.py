import torch
import tqdm
import json
import time
import asyncio
import os
import numpy as np
import torch.nn.functional as F
from importlib import import_module
from transformers import StoppingCriteria, set_seed




def load_hf_lm_and_tokenizer(
        args,
        device_map="auto", 
        torch_dtype="auto",
        load_in_8bit=False, 
        use_fast_tokenizer=True,
        padding_side="left",
    ):
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if args.metnohd == "ParamidKV":
        from models.modeling_llama_pyramid import LlamaForCausalLM as LlamaForCausalLM
    
     
    
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_flash_attention=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )
    if convert_to_half:
        model = model.half()
        
    
    model.eval()

        
    return model, tokenizer
