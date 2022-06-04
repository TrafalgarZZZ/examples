set -ex

python main_remote_openimages.py \
        -j $NUM_WORKER \
        -b $BATCH_SIZE \
        --epochs $EPOCH_NUM \
        -a $MODEL_ARCH \
        --server-host $SERVER_HOST \
        --dist-url "tcp://$MASTER_ADDR:$MASTER_PORT" \
        --dist-backend "nccl" \
        --multiprocessing-distributed \
        --world-size $WORLD_SIZE \
        --rank $RANK \
        /data/