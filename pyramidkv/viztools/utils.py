import os
import torch
import matplotlib.pyplot as plt


def plot_heatmap(
    data: torch.Tensor,
    title: str = "Attention Heatmap",
    fig_size: tuple[int, int] | None = None,
    x_label: str = "Key Positions",
    y_label: str = "Query Positions",
    show_ticks: bool = False,
    y_ticks: list[int] | None = None,
    cmap: str | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot a heatmap for a 2D tensor.

    Args:
        data (torch.Tensor): A 2D tensor to visualize as a heatmap.
        title (str): Title of the heatmap. Defaults to "Attention Heatmap".
        fig_size (tuple[int, int] | None): Size of the figure. If None, the default size is used.
        x_label (str): Label for the x-axis. Defaults to "Key Positions".
        y_label (str): Label for the y-axis. Defaults to "Query Positions".
        show_ticks (bool): Whether to show ticks on the heatmap. Defaults to False.
        y_ticks (list[int] | None): List of y-tick labels to display on the heatmap.
        cmap (str | None): Colormap to use for the heatmap. If None, the default colormap is used.
        save_path (str | None): File path to save the heatmap. If None, the heatmap is displayed but not saved.

    Returns:
        None.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_len, x_len = data.shape

    plt.figure(figsize=fig_size)
    plt.imshow(data, vmax=100, cmap=cmap, aspect="auto")

    if show_ticks:
        if x_len > 100:
            plt.xticks(ticks=range(0, x_len, 4), labels=range(0, x_len, 4), rotation=90, fontsize=4)
        else:
            plt.xticks(ticks=range(x_len), labels=range(x_len))
        plt.yticks(ticks=range(y_len), labels=y_ticks)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Attention heatmap saved to {save_path}")

    plt.show()


def attn_mp(
    attention_scores: torch.Tensor,
    high_value: float = 100,
    mid_value: float = 50
) -> torch.Tensor:
    """
    Normalize attention scores based on given conditions.

    Args:
        attention_scores (torch.Tensor): A tensor containing attention scores. The shape of the tensor is supposed to be (..., seq_len).
        high_value (float): The value to set for scores greater than or equal to 1 / seq_len. Defaults to 100.
        mid_value (float): The value to set for scores between 0 and high_value. Defaults to 50.

    Returns:
        torch.Tensor: The normalized attention scores.
    """
    seq_len = attention_scores.shape[-1]
    attention_scores[attention_scores >= 1 / seq_len] = high_value
    attention_scores[(attention_scores > 0) & (attention_scores < high_value)] = mid_value
    return attention_scores
