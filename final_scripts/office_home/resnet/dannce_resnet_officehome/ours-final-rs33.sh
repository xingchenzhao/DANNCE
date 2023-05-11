#!/bin/bash

random_seed=33

python3 -m src.main \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=officehome/dannce_resnet/art-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --adversarial_examples_lr=1e-1 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-2 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --single_target=art \
    --use_original_train_set \
    --entropy \
    --dataset=OfficeHome \
    --model=resnet \
    --domain_adversary_lr=1e-3 \
    --domain_adversary_weight=0.25

python3 -m src.main \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=officehome/dannce_resnet/clipart-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --adversarial_examples_lr=1e-1 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-2 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --single_target=clipart \
    --use_original_train_set \
    --entropy \
    --dataset=OfficeHome \
    --model=resnet \
    --domain_adversary_lr=1e-3 \
    --domain_adversary_weight=0.25

python3 -m src.main \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=officehome/dannce_resnet/product-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --adversarial_examples_lr=1e-1 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-2 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --single_target=product \
    --use_original_train_set \
    --entropy \
    --dataset=OfficeHome \
    --model=resnet \
    --domain_adversary_lr=1e-3 \
    --domain_adversary_weight=0.25

python3 -m src.main \
    --domain_adversary \
    --early_adversary_supression \
    --matsuura_augmentation \
    --adversarial_examples \
    --save_dir=officehome/dannce_resnet/realword-rs$((random_seed)) \
    --gpu=0 \
    --random_seed=$random_seed \
    --adversarial_examples_lr=1e-1 \
    --adversarial_train_steps=5 \
    --adversarial_examples_wd=1e-2 \
    --adversarial_examples_ratio=0.5 \
    --adv_blur_step=4 \
    --adv_kl_weight=1 \
    --single_target=real_world \
    --use_original_train_set \
    --entropy \
    --dataset=OfficeHome \
    --model=resnet \
    --domain_adversary_lr=1e-3 \
    --domain_adversary_weight=0.25