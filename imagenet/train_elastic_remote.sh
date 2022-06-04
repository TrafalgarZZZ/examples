python -m torchelastic.distributed.launch \
        --nnodes=1 \
        --nproc_per_node=2 \
        --rdzv_id=$JOB_UUID \
        --rdzv_backend=etcd \
        --rdzv_endpoint=192.168.180.12:2379 \
        main_elastic_remote.py \
        --arch resnet18 \
        --epochs 20 \
        --batch-size 32 \
        --workers 2 \
        /data/imagenet