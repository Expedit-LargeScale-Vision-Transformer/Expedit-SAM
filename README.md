# Expediting SAM without Fine-tuning

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2210.01035) 
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/kxqt/Expedit-SAM) 

## <a name="Introduction"></a>Introduction

This is the official implementation of the paper "[Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning](https://arxiv.org/abs/2210.01035)" on [Segment Anything Model (SAM)](https://segment-anything.com/).

![framework](assets/Hourglass_transformer_framework.png)
![framework](assets/TokenClusterReconstruct_Details.png)

Our method can speed up SAM without any training. The bottleneck of SAM is image encoder. We implement our method on image encoder to signifficantly speed up the generation process. We test our method on different SAM models using a single 16G Tesla-V100. We set `--points-per-side=12` and `--points-per-batch=144` so that the generation process executes only one time.

| Model            | clustering location | num of clusters | speed(image/s)     |
| ---------------- | ------------------- | --------------- | ------------------ |
| SAM-ViT-H        | -                   | -               | 1.27               |
| SAM-ViT-H + ours | 18                   | 121             | 1.40(1.10x faster) |
| SAM-ViT-H + ours | 14                   | 100             | 1.52(1.19x faster) |
| SAM-ViT-H + ours | 8                   | 100             | 1.64(1.30x faster) |
| SAM-ViT-H + ours | 8                   | 81              | 1.82(1.44x faster) |
| SAM-ViT-H + ours | 6                   | 81              | 1.89(1.49x faster) |

Here is the visualization of the setting above.

![result of sam-vit-h + ours](assets/result_vit_h.png)

We also try to implement our method on smaller model. Here are some examples generate by SAM w/ ViT-L + ours, with the setting of `--points-per-side=16` and `--points-per-batch=256`.  

![result of sam-vit-l + ours](assets/result_vit_l.png)

## <a name="Installation"></a>Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

To use Segment Anything with our method, please clone this repository locally and install with

```
pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```


## <a name="GettingStarted"></a>Getting Started

You can run the code like using original Segment Anything Model. The only difference is that you need to add `use_hourglass=True` as parameter while calling `build_sam` function. Here is an example.

First download a [model checkpoint](#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt:

```
from segment_anything import build_sam, SamPredictor 
predictor = SamPredictor(build_sam(checkpoint="</path/to/model.pth>", use_hourglass=True))
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from segment_anything import build_sam, SamAutomaticMaskGenerator
mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="</path/to/model.pth>", use_hourglass=True))
masks = mask_generator.generate(<your_image>)
```

Additionally, masks can be generated for images from the command line:

```
python scripts/amg.py --checkpoint <path/to/sam/checkpoint> --input <image_or_folder> --output <output_directory> --use_hourglass
```

You need to add `--use_hourglass` if you want to use our method to accelerate the process.


## <a name="Gradio Demo"></a>Gradio Demo

Web demo build with gradio is supported! You can run the following command to launch the demo:

```
python app.py
```

This demo is also hosted on HuggingFace [here](https://huggingface.co/spaces/kxqt/Expedit-SAM). Have fun!

## <a name="Models"></a>Model Checkpoints

Here are the official weight of SAM model.

* **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
* `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
* `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## License
The model is licensed under the [Apache 2.0 license](LICENSE).

## Citation

If you find this repo useful in your research, please consider citing:

```latex
@article{liang2022expediting,
	author    = {Liang, Weicong and Yuan, Yuhui and Ding, Henghui and Luo, Xiao and Lin, Weihong and Jia, Ding and Zhang, Zheng and Zhang, Chao and Hu, Han},
	title     = {Expediting large-scale vision transformer for dense prediction without fine-tuning},
	journal   = {arXiv preprint arXiv:2210.01035},
	year      = {2022},
}
```

If you use SAM or SA-1B in your research, please use the following BibTeX entry. 

```
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
