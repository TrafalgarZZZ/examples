set -ex

python main_imagenet.py \
        -j $NUM_WORKER \
        -b $BATCH_SIZE \
        --epochs $EPOCH_NUM \
        -a $MODEL_ARCH \
        --dist-url "tcp://$MASTER_ADDR:$MASTER_PORT" \
        --dist-backend "nccl" \
        --multiprocessing-distributed \
        --world-size $WORLD_SIZE \
        --rank $RANK \
        /data/imagenet/train /data/imagenet/val