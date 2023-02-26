# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
# from diffusion import create_diffusion
# from diffusers.models import AutoencoderKL
# from download import find_model
# from models import DiT_models
import argparse
import numpy as np
from PIL import Image
import os


# def main(args):
#     # Setup PyTorch:
#     torch.manual_seed(args.seed)
#     torch.set_grad_enabled(False)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     if args.ckpt is None:
#         assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
#         assert args.image_size in [256, 512]
#         assert args.num_classes == 1000

#     # Load model:
#     latent_size = args.image_size // 8
#     model = DiT_models[args.model](
#         input_size=latent_size,
#         num_classes=args.num_classes
#     ).to(device)
#     # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
#     ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
#     state_dict = find_model(ckpt_path)
#     model.load_state_dict(state_dict)
#     model.eval()  # important!
#     diffusion = create_diffusion(str(args.num_sampling_steps))
#     vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
#     using_cfg = args.cfg_scale > 1.0

#     # Create folder to save samples:
#     model_string_name = args.model.replace("/", "-")
#     ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
#     folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
#                   f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
#     sample_folder_dir = f"{args.sample_dir}/{folder_name}"
#     if rank == 0:
#         os.makedirs(sample_folder_dir, exist_ok=True)
#         print(f"Saving .png samples at {sample_folder_dir}")

#     for i in range(500):
#         n = 10
#         # Sample inputs:
#         z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
#         y = torch.randint(0, args.num_classes, (n,), device=device)

#         # Setup classifier-free guidance:
#         if using_cfg:
#             z = torch.cat([z, z], 0)
#             y_null = torch.tensor([1000] * n, device=device)
#             y = torch.cat([y, y_null], 0)
#             model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
#             sample_fn = model.forward_with_cfg
#         else:
#             model_kwargs = dict(y=y)
#             sample_fn = model.forward

#         # Sample images:
#         samples = diffusion.p_sample_loop(
#             sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
#         )
#         if using_cfg:
#             samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

#         samples = vae.decode(samples / 0.18215).sample
#         samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

#         # Save samples to disk as individual .png files
#         for j, sample in enumerate(samples):
#             index = i * 10 + j + total
#             Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
#         total += 10

#     # # Labels to condition the model with (feel free to change):
#     # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

#     # # Create sampling noise:
#     # n = len(class_labels)
#     # z = torch.randn(n, 4, latent_size, latent_size, device=device)
#     # y = torch.tensor(class_labels, device=device)

#     # # Setup classifier-free guidance:
#     # z = torch.cat([z, z], 0)
#     # y_null = torch.tensor([1000] * n, device=device)
#     # y = torch.cat([y, y_null], 0)
#     # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

#     # # Sample images:
#     # samples = diffusion.p_sample_loop(
#     #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    # # Save and display images:
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))

def transform(args):
    npz_file = np.load(args.npz_file)
    img_file = npz_file['arr_0']
    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.numbers):
        print(i)
        image = img_file[i]
        Image.fromarray(image).save(f"{args.out_dir}/{i:06d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_file", type=str, default="./boosted_diff_256_eps_3_steps_7_labeled.npz")
    parser.add_argument("--out_dir", type=str, default="/mnt/petrelfs/yangmengping/generate_data/ImageNet256/BigRoc/boosted_diff_256_eps_3_steps_7_labeled")
    parser.add_argument("--numbers", type=int, default=50000)
    args = parser.parse_args()
    transform(args)
