import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def infer_attention(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """
    Retrieve attention weights from a model and return them as a list of tensors.

    This function generates attention weights from a given model and tokenizer
    based on the provided prompt. The attention weights are extracted for each
    layer of the model and returned as a list of tensors.

    Args:
        model (AutoModelForCausalLM): The pretrained language model from which
            to extract attention weights.
        tokenizer (AutoTokenizer): The tokenizer used to encode the input prompt.
        prompt (str): The input text to generate attention weights for.
        device (str, optional): The device to run the model on. Defaults to "cpu".

    Returns:
        list[torch.Tensor]: A list of attention weight tensors, where each tensor
            has shape (num_heads, seq_len, seq_len), corresponding to the attention
            weights of each layer in the model.
    """
    model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attentions = model(input_ids, output_attentions=True).attentions

    return [attention.squeeze(0).detach().cpu() for attention in attentions]
