import logging
import math
import os
from PIL import Image
import torch
from torch.utils.data import RandomSampler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange
from diffusers import AutoencoderKLTemporalDecoder
from dataset import VideoDataset
from unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from controlnet_sdv import ControlNetSDVModel
from omegaconf import OmegaConf
from utils import get_rand_cosine_interpolated_samples
from diffusers.optimization import get_scheduler


logger = get_logger(__name__, log_level="INFO")


def get_images_from_folder(path_to_folder):
    imgs = []
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    def extract_frame_number(filename):
        parts = filename.split('.')
        try:
            return int(parts[0])
        except ValueError:
            return float('inf')

    ordered_files = sorted(os.listdir(path_to_folder), key=extract_frame_number)

    for fname in ordered_files:
        f_path = os.path.join(path_to_folder, fname)
        if os.path.splitext(fname)[1].lower() in supported_extensions:
            try:
                img = Image.open(f_path).convert('RGB')
                imgs.append(img)
            except Exception as e:
                print(f"Failed to load image {f_path}: {e}")

    depth_data_folder = path_to_folder + "_depth"
    depth_imgs = []
    if os.path.exists(depth_data_folder):
        ordered_depth_files = sorted(os.listdir(depth_data_folder), key=extract_frame_number)
        for fname in ordered_depth_files:
            f_path = os.path.join(depth_data_folder, fname)
            if os.path.splitext(fname)[1].lower() in supported_extensions:
                try:
                    img = Image.open(f_path).convert('RGB')
                    depth_imgs.append(img)
                except Exception as e:
                    print(f"Failed to load depth image {f_path}: {e}")
    else:
        print(f"Depth data folder not found: {depth_data_folder}")

    return imgs, depth_imgs


def convert_tensor_to_vae_latent(input_tensor, vae_model):
    num_frames = input_tensor.shape[1]

    input_tensor = rearrange(input_tensor, "b f c h w -> (b f) c h w")
    latents = vae_model.encode(input_tensor).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=num_frames)
    latents = latents * vae_model.config.scaling_factor

    return latents

def get_added_time_ids(
    frames_per_second,
    motion_bucket_identifiers,
    noise_augmentation_strength,
    data_type,
    batch_sz,
    device_target=None,
):
    target_dev = device_target if device_target is not None else 'cpu'

    if not isinstance(motion_bucket_identifiers, torch.Tensor):
        motion_bucket_identifiers = torch.tensor(motion_bucket_identifiers, dtype=data_type, device=target_dev)
    else:
        motion_bucket_identifiers = motion_bucket_identifiers.to(device=target_dev)

    if motion_bucket_identifiers.dim() == 1:
        motion_bucket_identifiers = motion_bucket_identifiers.view(-1, 1)

    if motion_bucket_identifiers.size(0) != batch_sz:
        raise ValueError("Motion bucket ID length must match batch size.")

    add_time_ids = torch.tensor([frames_per_second, noise_augmentation_strength], dtype=data_type, device=target_dev).repeat(batch_sz, 1)

    add_time_ids = torch.cat([add_time_ids, motion_bucket_identifiers], dim=1)

    return add_time_ids


def run_training_pipeline():
    cfg = OmegaConf.load("configs/train.yaml")

    log_directory = os.path.join(cfg.output_dir, cfg.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, logging_dir=log_directory)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        project_config=accelerator_project_config,
    )

    random_generator = torch.Generator(
        device=accelerator.device).manual_seed(23123134)


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    clip_feature_extractor = CLIPImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="feature_extractor", revision=cfg.revision)
    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_encoder", revision=cfg.revision, variant="fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision, variant="fp16")
    spatio_temporal_unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        cfg.pretrained_model_name_or_path if cfg.pretrain_unet is None else cfg.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    print("Loading ControlNet model...")
    controlnet_model: ControlNetSDVModel = ControlNetSDVModel.from_unet(spatio_temporal_unet)

    vae.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)
    spatio_temporal_unet.requires_grad_(False)

    precision_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        precision_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        precision_dtype = torch.bfloat16

    clip_image_encoder.to(accelerator.device, dtype=precision_dtype)
    vae.to(accelerator.device, dtype=precision_dtype)
    spatio_temporal_unet.to(accelerator.device, dtype=precision_dtype)


    controlnet_model.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, controlnet_model.parameters()),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    cfg.global_batch_size = cfg.per_gpu_batch_size * accelerator.num_processes

    train_dataset = VideoDataset(cfg.video_folder, min(cfg.width, cfg.height))
    data_sampler = RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=data_sampler,
        batch_size=cfg.per_gpu_batch_size,
        num_workers=cfg.num_workers,
    )

    override_max_train_steps = False
    num_update_steps_per_epoch_val = math.ceil(
        len(train_data_loader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch_val
        override_max_train_steps = True

    learning_rate_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
    )

    controlnet_model, optimizer, learning_rate_scheduler, train_data_loader = accelerator.prepare(
        controlnet_model, optimizer, learning_rate_scheduler, train_data_loader
    )

    num_update_steps_per_epoch_val = math.ceil(len(train_data_loader) / cfg.gradient_accumulation_steps)
    if override_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch_val
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch_val)

    current_global_step = 0
    start_epoch = 0

    if accelerator.is_main_process:
        trainable_params_count = sum(p.numel() for p in controlnet_model.parameters() if p.requires_grad)
        logger.info(f"Trainable ControlNet parameter count: {trainable_params_count:,}")


    def get_image_embeddings(pixel_vals):
        pixel_vals = pixel_vals * 2.0 - 1.0
        pixel_vals = torch.nn.functional.interpolate(pixel_vals, size=(224, 224), mode="bicubic", align_corners=True, antialias=True)
        pixel_vals = (pixel_vals + 1.0) / 2.0

        pixel_vals = clip_feature_extractor(
            images=pixel_vals.cpu().numpy(),
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_vals = pixel_vals.to(
            device=accelerator.device, dtype=precision_dtype)
        img_embeddings = clip_image_encoder(pixel_vals).image_embeds
        img_embeddings= img_embeddings.unsqueeze(1)
        return img_embeddings

    progress_bar = tqdm(range(current_global_step, cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch_idx in range(start_epoch, cfg.num_train_epochs):
        controlnet_model.train()
        current_train_loss = 0.0
        for step_idx, batch_data in enumerate(train_data_loader):
            with accelerator.accumulate(controlnet_model):
                pixel_values_batch = batch_data["pixel_values"].to(precision_dtype).to(
                    accelerator.device, non_blocking=True
                )
                latents_batch = convert_tensor_to_vae_latent(pixel_values_batch, vae)

                noise_tensor = torch.randn_like(latents_batch)
                batch_size_val = latents_batch.shape[0]
                sigmas_sampled = get_rand_cosine_interpolated_samples(shape=[batch_size_val,], image_d=64, noise_d_low=32, noise_d_high=64,
                                                  sigma_data=0.5, min_value=0.002, max_value=700).to(latents_batch.device)
                sigmas_reshaped = sigmas_sampled.clone()
                while len(sigmas_reshaped.shape) < len(latents_batch.shape):
                    sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)

                training_noise_aug_val = 0.02
                small_noise_latents_batch = latents_batch + noise_tensor * training_noise_aug_val
                conditional_latents_batch = small_noise_latents_batch[:, 0, :, :, :]
                conditional_latents_batch = conditional_latents_batch / vae.config.scaling_factor


                noisy_latents_batch  = latents_batch + noise_tensor * sigmas_reshaped
                timesteps_batch = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas_sampled]).to(latents_batch.device)


                input_noisy_latents_batch = noisy_latents_batch  / ((sigmas_reshaped**2 + 1) ** 0.5)


                encoder_hidden_states_batch = get_image_embeddings(
                    pixel_values_batch[:, 0, :, :, :])

                added_time_ids_batch = get_added_time_ids(
                    6,
                    batch_data["motion_values"],
                    training_noise_aug_val,
                    encoder_hidden_states_batch.dtype,
                    batch_size_val,
                    spatio_temporal_unet,
                    device_target=latents_batch.device
                )
                added_time_ids_batch = added_time_ids_batch.to(latents_batch.device)

                if cfg.conditioning_dropout_prob is not None:
                    rp = torch.rand(batch_size_val, device=latents_batch.device, generator=random_generator)
                    prompt_mask_val = rp < 2 * cfg.conditioning_dropout_prob
                    prompt_mask_val = prompt_mask_val.reshape(batch_size_val, 1, 1)
                    null_conditioning_val = torch.zeros_like(encoder_hidden_states_batch)
                    encoder_hidden_states_batch = torch.where(prompt_mask_val, null_conditioning_val, encoder_hidden_states_batch)

                    image_mask_data_type = conditional_latents_batch.dtype
                    image_mask_val = 1 - ((rp >= cfg.conditioning_dropout_prob).to(image_mask_data_type)* (rp < 3 * cfg.conditioning_dropout_prob).to(image_mask_data_type))
                    image_mask_val = image_mask_val.reshape(batch_size_val, 1, 1, 1)
                    conditional_latents_batch = image_mask_val * conditional_latents_batch

                conditional_latents_batch = conditional_latents_batch.unsqueeze(1).repeat(1, noisy_latents_batch.shape[1], 1, 1, 1)
                input_noisy_latents_batch = torch.cat([input_noisy_latents_batch, conditional_latents_batch], dim=2)
                controlnet_input_image = batch_data["canny"]

                target_latents = latents_batch

                down_block_residuals, mid_block_residual = controlnet_model(
                    input_noisy_latents_batch, timesteps_batch, encoder_hidden_states_batch,
                    added_time_ids=added_time_ids_batch,
                    controlnet_cond=controlnet_input_image,
                    return_dict=False,
                )

                model_predicted_output = spatio_temporal_unet(
                    input_noisy_latents_batch, timesteps_batch, encoder_hidden_states_batch,
                    added_time_ids=added_time_ids_batch,
                    down_block_additional_residuals=[
                        sample_res.to(dtype=precision_dtype) for sample_res in down_block_residuals
                    ],
                    mid_block_additional_residual=mid_block_residual.to(dtype=precision_dtype),
                ).sample

                sigmas_final = sigmas_reshaped
                c_out_val = -sigmas_final / ((sigmas_final**2 + 1)**0.5)
                c_skip_val = 1 / (sigmas_final**2 + 1)
                denoised_latents_output = model_predicted_output * c_out_val + c_skip_val * noisy_latents_batch
                weighting_factor = (1 + sigmas_final ** 2) * (sigmas_final**-2.0)

                current_loss = torch.mean((weighting_factor.float() * (denoised_latents_output.float() - target_latents.float()) ** 2).reshape(target_latents.shape[0], -1), dim=1)
                current_loss = current_loss.mean()

                avg_loss_val = accelerator.gather(
                    current_loss.repeat(cfg.per_gpu_batch_size)).mean()
                current_train_loss += avg_loss_val.item() / cfg.gradient_accumulation_steps

                accelerator.backward(current_loss)
                optimizer.step()
                learning_rate_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                current_global_step += 1
                accelerator.log({"train_loss": current_train_loss}, step=current_global_step)
                current_train_loss = 0.0

                if accelerator.is_main_process:
                    if current_global_step % cfg.checkpointing_steps == 0:
                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{current_global_step}")
                        accelerator.unwrap_model(controlnet_model).save_pretrained(os.path.join(save_path, "controlnet"))

                        logger.info(f"Checkpoint saved to {save_path}")

            log_info = {"step_loss": current_loss.detach().item(), "lr": learning_rate_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**log_info)

            if current_global_step >= cfg.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet_model = accelerator.unwrap_model(controlnet_model)
        controlnet_model.save_pretrained(os.path.join(cfg.output_dir, "controlnet"))

    accelerator.end_training()


if __name__ == "__main__":
    run_training_pipeline()