from pathlib import Path
from argparse import Namespace
import re
from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2

from svd_controlnet_pipeline import StableVideoDiffusionPipelineControlNet
from controlnet_sdv import ControlNetSDVModel
from unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel


def extract_frame_number(filename: str) -> int:
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


def load_and_process_images(folder: Path, target_size: Tuple[int, int] = (512, 512)) -> List[Image.Image]:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    image_paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in valid_extensions],
        key=lambda x: extract_frame_number(x.name)
    )

    images = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(Image.fromarray(img))

    return images


if __name__ == "__main__":
    args = Namespace(
        pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid",
        validation_image_folder=Path("./test/img"),
        validation_control_folder=Path("./test/depth"),
        validation_image=Path("./test/test.png"),
        output_dir=Path("./output"),
        height=512,
        width=512
    )

    validation_images = load_and_process_images(args.validation_image_folder, target_size=(args.width, args.height))
    validation_control_images = load_and_process_images(args.validation_control_folder, target_size=(args.width, args.height))
    validation_image = Image.open(args.validation_image).convert("RGB")

    controlnet = ControlNetSDVModel.from_pretrained("controlnet", subfolder="controlnet")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(args.pretrained_model_name_or_path, controlnet=controlnet, unet=unet)
    pipeline.enable_model_cpu_offload()

    val_save_dir = args.output_dir / "validation_images"
    val_save_dir.mkdir(parents=True, exist_ok=True)

    video_frames = pipeline(
        validation_image,
        validation_control_images[:14],
        decode_chunk_size=8,
        num_frames=14,
        motion_bucket_id=10,
        controlnet_cond_scale=1.0,
        width=args.width,
        height=args.height
    ).frames

    for i, frame in enumerate(video_frames):
        frame.save(val_save_dir / f"frame_{i:03d}.png")
