#!/bin/bash

random_seed=0

for i in {1..3}
do
    ((random_seed=random_seed+15+i))
    python3 -m src.main \
        --matsuura_augmentation \
        --save_dir=pacs/deepall-alexnet/rs$((random_seed))_run$i \
        --gpu=0 \
        --random_seed=$random_seed \
        --use_original_train_set
done