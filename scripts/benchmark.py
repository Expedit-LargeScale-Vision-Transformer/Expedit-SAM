# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.amg import (
    batch_iterator,
    generate_crop_boxes,
)

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    default="default",
    help="The type of model to load, in ['default', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)

# add hourglass settings
amg_settings.add_argument(
    "--use_hourglass",
    action="store_true",
    help="Use hourglass method to expedite mask generation.",
)

amg_settings.add_argument(
    "--hourglass_clustering_location",
    type=int,
    default=6,
    help="location of clustering, ranging from [0, num of layers of transformer)"
)

amg_settings.add_argument(
    "--hourglass_num_cluster",
    type=int,
    default=100,
    help="num of clusters, no more than total number of features"
)

amg_settings.add_argument(
    "--hourglass_cluster_iters",
    type=int,
    default=5,
    help="num of iterations in clustering"
)

amg_settings.add_argument(
    "--hourglass_temperture",
    type=float,
    default=5e-3,
    help="temperture in clustering and reconstruction"
)

amg_settings.add_argument(
    "--hourglass_cluster_window_size",
    type=int,
    default=5,
    help="window size in clustering"
)

amg_settings.add_argument(
    "--hourglass_reconstruction_k",
    type=int,
    default=20,
    help="k in token reconstruction layer of hourglass vit"
)

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def get_hourglass_kwargs(args):
    hourglass_kwargs = {
        "use_hourglass": args.use_hourglass,
        "hourglass_clustering_location": args.hourglass_clustering_location,
        "hourglass_num_cluster": args.hourglass_num_cluster,
        "hourglass_cluster_iters": args.hourglass_cluster_iters,
        "hourglass_temperture": args.hourglass_temperture,
        "hourglass_cluster_window_size": args.hourglass_cluster_window_size,
        "hourglass_reconstruction_k": args.hourglass_reconstruction_k,
    }
    hourglass_kwargs = {k: v for k, v in hourglass_kwargs.items() if v is not None}
    return hourglass_kwargs


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    hourglass_kwargs = get_hourglass_kwargs(args)
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint, **hourglass_kwargs)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    total_time = 0
    warmup = 50
    num_samples = 200
    for i in tqdm(range(num_samples)):
        image = np.random.randint(0, 255, size=(1024, 1024, 3), dtype=np.uint8)

        start = time.perf_counter()
        # masks = generator.generate(image)
        with torch.no_grad():
            # mask_data = generator._generate_masks(image)
            orig_size = image.shape[:2]
            crop_boxes, layer_idxs = generate_crop_boxes(
                orig_size, generator.crop_n_layers, generator.crop_overlap_ratio
            )

            # Iterate over image crops
            for crop_box, crop_layer_idx in zip(crop_boxes, layer_idxs):
                # crop_data = generator._process_crop(image, crop_box, layer_idx, orig_size)
                x0, y0, x1, y1 = crop_box
                cropped_im = image[y0:y1, x0:x1, :]
                cropped_im_size = cropped_im.shape[:2]
                generator.predictor.set_image(cropped_im)

                points_scale = np.array(cropped_im_size)[None, ::-1]
                points_for_image = generator.point_grids[crop_layer_idx] * points_scale

                for (points,) in batch_iterator(generator.points_per_batch, points_for_image):
                    # batch_data = generator._process_batch(points, cropped_im_size, crop_box, orig_size)
                    transformed_points = generator.predictor.transform.apply_coords(points, cropped_im_size)
                    in_points = torch.as_tensor(transformed_points, device=generator.predictor.device)
                    in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
                    masks, iou_preds, _ = generator.predictor.predict_torch(
                        in_points[:, None, :],
                        in_labels[:, None],
                        multimask_output=True,
                        return_logits=True,
                    )
                    del masks
                    del iou_preds

        eta = time.perf_counter() - start
        if i >= warmup:
            total_time += eta
    print("Done!")
    print(f"Average time per image: {total_time / (num_samples - warmup)} seconds")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
