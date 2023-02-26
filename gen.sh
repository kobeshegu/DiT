srun -p digitalcity -N1 --quotatype=reserved --job-name=DiT --gres=gpu:1 --cpus-per-task=16 \
    torchrun --nnodes=1 --nproc_per_node=1 sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000 \