#!/bin/bash

random_seed=33

python3 -m src.main \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=ablate_blur/dann-pacs-alexnet-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --adversarial_examples_lr=1e-3 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-3 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --use_original_train_set \
    --entropy \
    --ablate_blur

