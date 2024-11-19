import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from packaging import version
from hqq.core.quantize import Quantizer as HQQQuantizer

from transformers.cache_utils import QuantizedCache,CacheConfig
class KVQuantizedCache(QuantizedCache):
    def __init__(self, cache_config, outlier_threshold=6.0):
        super().__init__(cache_config)
        if self.nbits not in [1, 2, 3, 4, 8]:
            raise ValueError(
                f"`nbits` for `KIVI` backend has to be one of [`1`, `2`, `3`, `4`, `8`] but got {self.nbits}"
            )

        if self.axis_key not in [0, 1]:
            raise ValueError(f"`axis_key` for `KIVI` backend has to be one of [`0`, `1`] but got {self.axis_key}")

        if self.axis_value not in [0, 1]:
            raise ValueError(f"`axis_value` for `KIVI` backend has to be one of [`0`, `1`] but got {self.axis_value}")
        print("kvquant")
        self.quantizer = HQQQuantizer
        self.outlier_threshold = outlier_threshold

    def _quantize(self, tensor, axis):
        tensor, outlier_indices, outlier_values = self._handle_outliers(tensor)
        qtensor, meta = self.quantizer.quantize(
            tensor,
            axis=axis,
            device=self.device,
            compute_dtype=self.compute_dtype,
            nbits=self.nbits,
            group_size=self.q_group_size,
        )
        meta["compute_dtype"] = self.compute_dtype
        self.quantizer.cuda(qtensor, meta=meta, device=self.device)  # Move to device and cast to dtype
        return qtensor, meta, outlier_indices, outlier_values
        
    def _dequantize(self, qtensor, meta, outlier_indices, outlier_values):

        tensor = self.quantizer.dequantize(qtensor, meta)
        outlier_indices = torch.stack(outlier_indices, dim=-1)
        
        tensor[outlier_indices.unbind(dim=1)] = outlier_values
        return tensor

    def _handle_outliers(self, tensor):
        lower_bound = -self.outlier_threshold
        upper_bound = self.outlier_threshold
        outliers = (tensor < lower_bound) | (tensor > upper_bound)
        outlier_indices = torch.nonzero(outliers, as_tuple=True)
        outlier_values = tensor[outliers]
        tensor = tensor.masked_fill(outliers, 0)
        return tensor, outlier_indices, outlier_values
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    )-> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if len(self.key_cache) <= layer_idx:
            qt_key, meta_key, outlier_indices_key, outlier_values_key = self._quantize(key_states.contiguous(), axis=self.axis_key)
            qt_value, meta_value, outlier_indices_value, outlier_values_value = self._quantize(value_states.contiguous(), axis=self.axis_value)
            self._quantized_key_cache.append((qt_key, meta_key, outlier_indices_key, outlier_values_key))
            self._quantized_value_cache.append((qt_value, meta_value, outlier_indices_value, outlier_values_value))
            self.key_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            self.value_cache.append(torch.zeros(0, dtype=key_states.dtype, device=key_states.device))
            keys_to_return, values_to_return = key_states, value_states
        else:
            dequant_key = self._dequantize(*self._quantized_key_cache[layer_idx])
            dequant_value = self._dequantize(*self._quantized_value_cache[layer_idx])
            keys_to_return = [dequant_key, self.key_cache[layer_idx], key_states]
            values_to_return = [dequant_value, self.value_cache[layer_idx], value_states]
            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)
            if (
                self.key_cache[layer_idx].dim() == 4
                and self.key_cache[layer_idx].shape[-2] + 1 >= self.residual_length
            ):
                self._quantized_key_cache[layer_idx] = self._quantize(keys_to_return.contiguous(), axis=self.axis_key)
                self._quantized_value_cache[layer_idx] = self._quantize(
                    values_to_return.contiguous(), axis=self.axis_value
                )
                self.key_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
                self.value_cache[layer_idx] = torch.zeros(0, dtype=key_states.dtype, device=key_states.device)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return keys_to_return, values_to_return

