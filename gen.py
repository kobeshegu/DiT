"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time
import logging

import numpy as np
import torch.distributed as dist

import torch
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2
import os

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--out_dir', default='/mnt/petrelfs/yangmengping/generate_data/ImageNet256/DiT')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--resume", action='store_true')
    args = parser.parse_args()
    print(args)
    # init ddp
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    # Setup PyTorch:
    rank = torch.distributed.get_rank()
    logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
    if args.resume:
        seed = int(time.time())
        torch.manual_seed(seed + rank)
    else:
        torch.manual_seed(0 + rank)
    torch.set_grad_enabled(False)
    device = torch.device("cuda", local_rank)
    num_sampling_steps = 250
    cfg_scale = 1.5

    # Load model:
    image_size = args.image_size
    assert image_size in [256, 512], "We only provide pre-trained models for 256x256 and 512x512 resolutions."
    latent_size = image_size // 8
    model = DiT_XL_2(input_size=latent_size).to(device)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(num_sampling_steps))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    shape_str = f"{args.num_samples}x{image_size}x{image_size}x3"
    out_path = os.path.join(args.out_dir, f"samples_{shape_str}.npz")

    logging.info("sampling...")
    generated_num = torch.tensor(0, device=device)
    if rank == 0:
        all_images = []
        all_labels = []
        if args.resume:
            if os.path.exists(out_path):
                ckpt = np.load(out_path)
                all_images = ckpt['arr_0']
                all_labels = ckpt['arr_1']
                assert all_images.shape[0] % args.batch_size == 0, f'Wrong resume checkpoint shape {all_images.shape}'
                all_images = np.split(all_images,
                                      all_images.shape[0] // args.batch_size,
                                      0)
                all_labels = np.split(all_labels,
                                      all_labels.shape[0] // args.batch_size,
                                      0)

                logging.info('successfully resume from the ckpt')
                logging.info(f'Current number of created samples: {len(all_images) * args.batch_size}')
        generated_num = torch.tensor(len(all_images) * args.batch_size, device=device)
    dist.barrier()
    dist.broadcast(generated_num, 0)

    while generated_num.item() < args.num_samples:
        class_labels = torch.randint(low=0,
                                     high=args.num_classes,
                                     size=(args.batch_size,),
                                     device=device)
        n = class_labels.size(0)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = class_labels

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)

        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        samples.clamp_(min=-1, max=1)
        samples.sub_(-1).div_(2)
        samples = samples.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        samples = samples.permute(0, 2, 3, 1)
        samples = samples.contiguous()

        gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, samples)  # gather not supported with NCCL

        gathered_labels = [
            torch.zeros_like(class_labels) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_labels, class_labels)

        if rank == 0:
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logging.info(f"created {len(all_images) * args.batch_size} samples")
            generated_num = torch.tensor(len(all_images) * args.batch_size, device=device)
            if args.resume:
                if generated_num % 1000 == 0:
                    arr = np.concatenate(all_images, axis=0)
                    arr = arr[: args.num_samples]

                    label_arr = np.concatenate(all_labels, axis=0)
                    label_arr = label_arr[: args.num_samples]
                    logging.info(f"intermediate results saved to {out_path}")
                    np.savez(out_path, arr, label_arr)
                    del arr
                    del label_arr
        torch.distributed.barrier()
        dist.broadcast(generated_num, 0)

    if rank == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]

        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

        logging.info(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logging.info("sampling complete")


if __name__ == "__main__":
    main()