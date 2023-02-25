srun -p digitalcity -N1 --quotatype=reserved --job-name=DiT --gres=gpu:1 --cpus-per-task=16 \
    python gen.py