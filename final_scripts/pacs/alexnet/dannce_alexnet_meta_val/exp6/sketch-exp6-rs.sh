#!/bin/bash

random_seed=0
for i in {1..3}
do
    ((random_seed=random_seed+15+i))
    python3 -m src.main \
        --domain_adversary \
        --early_adversary_supression \
        --matsuura_augmentation \
        --adversarial_examples \
        --save_dir=dannce_alexnet_pacs_meta_val/sketch-exp6-rs$((random_seed)) \
        --gpu=0 \
        --random_seed=$random_seed \
        --adversarial_examples_lr=1e-1 \
        --adversarial_train_steps=5 \
        --adversarial_examples_wd=1e-2 \
        --adversarial_examples_ratio=0.5 \
        --adv_blur_step=4 \
        --adv_kl_weight=1 \
        --leave_out_domain=sketch \
        --use_original_train_set \
        --entropy
done