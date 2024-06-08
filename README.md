# Pyramid KV


Official repository for the paper "[PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling](https://arxiv.org/pdf/2406.02069)".

<p align="center">
    <img src="figs/PyramidKV.png" width="90%"> <br>
</p>

## Performence

<p align="center">
    <img src="figs/Result.png" width="90%"> <br>
</p>


## Requirements & Installation

```python

git clone https://github.com/Zefan-Cai/PyramidKV.git
cd PyramidKV
pip install -r requirements.txt .

```

## Inference

The checkpoint of model should be placed at: 'PyramidKV/ckpt'


```markdown
PyramidKV
    |- Pyramidkv
    |- data
    |- ckpt
        |- Llama-3-8B-Instruct
    |- ...
```


We support inference code on 'LongBench' to repuduce our result.

Please refer to 'scripts/scripts_longBench/eval.sh' to modify the parameters according to your requirements.

* CUDA_VISIBLE_DEVICES: LLaMA3 inference support on single GPU.

* base_dir: Set it to the path of PyramidKV. (e.g. /home/PyramidKV)

* model_name: Support "Llama-3-8B-Instruct"

* method: Support "PyramidKV"

* max_capacity_prompts: Selected KV Size in each layer. （e.g. 128, 2048 in paper）. When method is "PyramidKV", given that the total number of KV remains unchanged, the specific KV length for each layer will be modified accordingly

* save_dir: Path to your dir to save LongBench result.




```bash 

sh scripts/scripts_longBench/eval.sh

```

## Citation

If you find **PyramidKV** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@misc{cai2024pyramidkv,
      title={PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling}, 
      author={Zefan Cai. and Yichi Zhang and Bofei Gao and Tianyu Liu and Keming Lu and Wayne Xiong and Yue Dong and Baobao Chang and Junjie Hu and Wen Xiao},
      year={2024},
      eprint={2406.02069},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement


Thanks **[SnapKV]** [SnapKV: LLM Knows What You are Looking for Before Generation](https://github.com/FasterDecoding/SnapKV) for providing open-source code to support the expansion of this project's code.