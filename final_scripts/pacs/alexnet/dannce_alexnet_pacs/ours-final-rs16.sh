#!/bin/bash

random_seed=16

python3 -m src.main \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=pacs/dannce-alexnet/art-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --adversarial_examples_lr=1e-3 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-3 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --single_target=art_painting \
    --use_original_train_set \
    --entropy

python3 -m src.main \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=pacs/dannce-alexnet/sketch-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --adversarial_examples_lr=1e-2 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-2 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --single_target=sketch \
    --use_original_train_set \
    --entropy

python3 -m src.main \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=pacs/dannce-alexnet/cartoon-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --adversarial_examples_lr=1e-3 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-2 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --single_target=cartoon \
    --use_original_train_set \
    --entropy

python3 -m src.main \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=pacs/dannce-alexnet/photo-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --wandb=AISTATS_matsuuraSetup \
    --adversarial_examples_lr=1e-1 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-2 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --single_target=photo \
    --use_original_train_set \
    --entropy