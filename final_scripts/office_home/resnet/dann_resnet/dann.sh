#!/bin/bash

random_seed=0

for i in {1..3}
do
    ((random_seed=random_seed+15+i))
    python3 -m src.main \
        --early_adversary_supression \
        --matsuura_augmentation \
        --entropy \
        --domain_adversary \
        --save_dir=officehome/dann-resnet/rs$((random_seed))_run$i \
        --gpu=0 \
        --random_seed=$random_seed \
        --use_original_train_set \
        --model=resnet \
        --dataset=OfficeHome \
        --domain_adversary_lr=1e-3 \
        --domain_adversary_weight=0.25
done
