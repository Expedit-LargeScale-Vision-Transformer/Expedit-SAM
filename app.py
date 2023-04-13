import os
import time
import torch
import numpy as np

import gradio as gr

from segment_anything import build_sam, SamAutomaticMaskGenerator
from segment_anything.utils.amg import (
    batch_iterator,
    MaskData,
    calculate_stability_score,
    batched_mask_to_box,
    is_box_near_crop_edge,
)

os.system(r'python -m wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')

hourglass_args = {
    "baseline": {},
    "1.2x faster": {
        "use_hourglass": True,
        "hourglass_clustering_location": 14,
        "hourglass_num_cluster": 100,
    },
    "1.5x faster": {
        "use_hourglass": True,
        "hourglass_clustering_location": 6,
        "hourglass_num_cluster": 81,
    },
}

def predict(image, speed_mode, points_per_side):
    points_per_side = int(points_per_side)
    mask_generator = SamAutomaticMaskGenerator(
        build_sam(checkpoint="sam_vit_h_4b8939.pth", **hourglass_args[speed_mode]), 
        points_per_side=points_per_side,
        points_per_batch=64 if points_per_side > 12 else points_per_side * points_per_side
    )
    start = time.perf_counter()
    with torch.no_grad():
        masks = mask_generator.generate(image)
    eta = time.perf_counter() - start
    eta_text = f"Time of generation: {eta:.2f} seconds"

    if len(masks) == 0:
        return image
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    img = np.ones(image.shape)
    for mask in sorted_masks:
        m = mask['segmentation']
        color_mask = np.random.random((1, 1, 3))
        img = img * (1 - m[..., None]) + color_mask * m[..., None]

    image = ((image + img * 255) / 2).astype(np.uint8)
    return image, eta_text

description = """
#  <center>Expedit-SAM (Expedite Segment Anything Model without any training)</center>
Github link: [Link](https://github.com/Expedit-LargeScale-Vision-Transformer/Expedit-SAM)
You can select the speed mode you want to use from the "Speed Mode" dropdown menu and click "Run" to segment the image you uploaded to the "Input Image" box.
"""
if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    description += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'


def main():
    with gr.Blocks() as demo:
        gr.Markdown(description)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input Image")
                    with gr.Row():
                        points_per_side = gr.Dropdown(
                            choices=[4, 6, 8, 12, 16, 32],
                            value=12, 
                            label="Points per Side",
                        )
                        speed_mode = gr.Dropdown(
                            choices=list(hourglass_args.keys()),
                            value="baseline", 
                            label="Speed Mode",
                            multiselect=False,
                        )
                    with gr.Row():
                        run_btn = gr.Button(label="Run", id="run", value="Run")
                        clear_btn = gr.Button(label="Clear", id="clear", value="Clear")
                with gr.Column():
                    output_image = gr.Image(label="Output Image")
                    eta_label = gr.Label(label="ETA")
            gr.Examples(
                examples=[
                    ["./notebooks/images/dog.jpg"],
                    ["notebooks/images/groceries.jpg"],
                    ["notebooks/images/truck.jpg"],
                ],
                inputs=[input_image],
                outputs=[output_image],
                fn=predict,
            )
        
        run_btn.click(
            fn=predict, 
            inputs=[input_image, speed_mode, points_per_side], 
            outputs=[output_image, eta_label]
        )
        clear_btn.click(
            fn=lambda: [None, None], 
            inputs=None, 
            outputs=[input_image, output_image], 
            queue=False,
        )

    demo.queue()
    demo.launch()

if __name__ == "__main__":
    main()
