from transformers.models.llama.modeling_llama import *
from minference.modules.minference_forward import minference_prefill_forward
from minference.configs.model2path import MODEL2PATH
import json
from transformers.modeling_flash_attention_utils import _flash_attention_forward

minference_config = None

def init_minference(model_name):
    config_path = MODEL2PATH[model_name]
    global minference_config
    minference_config = json.load(open(config_path))

def minference_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    assert output_attentions == False, "output_attentions is not supported for MInference"

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    if q_len != 1: # prefill
        minference_kwards = {
            "layer_idx": self.layer_idx,
            "attn_forward_config": {"best_pattern": minference_config},
        }
        attn_output = minference_prefill_forward(
            query_states,
            key_states,
            value_states,
            {"attn_forward_config": minference_config, "layer_idx": self.layer_idx},
        )
    else:
        attn_output = _flash_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=self.attention_dropout,
            sliding_window=getattr(self, "sliding_window", None),
            is_causal=self.is_causal,
        )

    assert attn_output.size(1) == q_len
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
