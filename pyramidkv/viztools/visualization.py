import os
import torch
from typing import Callable
from .utils import plot_heatmap


OUTPUT_DIR = "../obs"

def plot_attention_heatmaps(
    attentions: list[torch.Tensor],
    map_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    layer_ids: list[int] | None = None,
    head_ids: list[int] | None = None,
    save_dir: str | None = "",
) -> None:
    """
    Visualize attention weights as heatmaps.

    This function generates heatmaps for specified layers and heads using a list of attention weight tensors, where each tensor has shape (num_heads, seq_len, seq_len).

    Args:
        attentions (list[torch.Tensor]): A list of attention weight tensors of shape (num_heads, seq_len, seq_len).
        map_fn (Callable[[torch.Tensor], torch.Tensor] | None): A function to apply to each attention tensor before visualization.
        layer_ids (list[int] | None): A list of indices of layers to visualize.
            If None, all layers are included.
        head_ids (list[int] | None): A list of indices of heads to visualize.
            If None, attention weights are averaged across all heads.
        save_dir (str | None): Directory to save the heatmaps. If None, the
            heatmaps are not saved to disk.

    Returns:
        None.
    """
    assert layer_ids is not None, "Please provide the layer_ids to visualize."

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if map_fn:
        attentions = [map_fn(attention) for attention in attentions]

    for layer_id in layer_ids:
        attention = attentions[layer_id]
        if not head_ids:
            data = torch.mean(attention, dim=0)
            save_path = os.path.join(OUTPUT_DIR, save_dir, f"layer{layer_id}.jpg") if save_dir else None
            plot_heatmap(data, title=f'Average Attention Map: Layer {layer_id}', save_path=save_path)
        else:
            for head_id in head_ids:
                data = attention[head_id]
                save_path = os.path.join(OUTPUT_DIR, save_dir, f"layer{layer_id}_head{head_id}.jpg") if save_dir else None
                plot_heatmap(data, title=f'Attention Map: Layer {layer_id} Head {head_id}', save_path=save_path)
