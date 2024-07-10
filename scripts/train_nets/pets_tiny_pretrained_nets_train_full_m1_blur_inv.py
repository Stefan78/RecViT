import os
import sys
import torch.optim as optim
import timm
import torch
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from pckgs.data import datasets
from pckgs.networks.utils import training
from config.definitions import NETWORKS_DIR
from pckgs.networks.recurrent_vision_transformer import *


hp_defaults = dict(
    lr=0.000133,
    warmup=0,
    batch_size=16,
    lr_decay=0.75,
    n_epochs=10,
    train_option=1,
    weight_decay=0.001,
    im_size=224,
)

for n_loops in [4]:
    if n_loops >= 4:    # due to limited GPU
        hp_defaults['batch_size'] = 8
    for n_repeat in range(5):
        wandb.init(config=hp_defaults, project="RViT_Experiments", group='pet_full_tiny_pretrained_5x5_blurinv_m1_final')
        config = wandb.config

        cuda_idx = 0
        model = timm.create_model("recurrent_vit_tiny_patch16_224", pretrained=True, patch_size=8)
        model.head = torch.nn.Linear(model.head.in_features, 37)

        train_loader, test_loader, inp_shape, classes = datasets.load_data('PET',
                                                                           batch_size_train=config['batch_size'],
                                                                           batch_size_test=config['batch_size'],
                                                                           transform="imagenet_statistics",
                                                                           imsize=config['im_size'])

        print(f'No. of parameters: {sum(p.numel() for p in model.parameters())}')

        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = training.CustomOptimizer(optimizer, default_lr=config['lr'], lr_decay=config['lr_decay'],
                                             warmup=(True, config['warmup']), lr_multiplier=1)

        device = f'cuda:{cuda_idx}' if torch.cuda.is_available() else "cpu"
        print(f"Training on {device}")

        # create folder if it does not exist
        folder = NETWORKS_DIR + '/PET'
        if not os.path.exists(folder):
            os.makedirs(folder)
        name = folder + f'/pet_tiny_pretrained_k_{n_loops}_run_{n_repeat}_blurinv'

        training.train_and_test_rvit(model, train_loader, test_loader, optimizer, device, epochs=config['n_epochs'],
                                     save=True, name=name, criterion=criterion, wandb_log=True,
                                     scheduler=scheduler, n_loops=n_loops, train_option=config['train_option'],
                                     random_transform=False, blur=True, blurinv=True)
        wandb.finish()
