import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from pckgs.data import datasets
from pckgs.networks.utils import training
import torch.optim as optim
from config.definitions import NETWORKS_DIR
from pckgs.networks.recurrent_vision_transformer import *
import timm
import torch
import wandb

hp_defaults = dict(
    lr=0.00007,
    warmup=0,
    batch_size=64,
    lr_decay=0.84,
    n_epochs=15,
    train_option=1,
    weight_decay=0.02,
    im_size=224,
)

for n_loops in [1]:
    for n_repeat in range(5):
        wandb.init(config=hp_defaults, project="RViT_Experiments", group='CIF10_5x5_baseline_RViT_m1')
        config = wandb.config

        cuda_idx = 0
        model = timm.create_model("recurrent_vit_tiny_patch16_224", pretrained=True)
        model.head = torch.nn.Linear(model.head.in_features, 10)

        train_loader, test_loader, inp_shape, classes = datasets.load_data('CIFAR10',
                                                                           batch_size_train=config['batch_size'],
                                                                           batch_size_test=config['batch_size'],
                                                                           transform="special_2",
                                                                           imsize=config['im_size'])

        print(f'No. of parameters: {sum(p.numel() for p in model.parameters())}')
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = training.CustomOptimizer(optimizer, default_lr=config['lr'], lr_decay=config['lr_decay'],
                                             warmup=(True, config['warmup']), lr_multiplier=1)

        device = f'cuda:{cuda_idx}' if torch.cuda.is_available() else "cpu"
        print(f"Training on {device}")

        # create folder if it does not exist
        folder = NETWORKS_DIR + '/CIFAR_10'
        if not os.path.exists(folder):
            os.makedirs(folder)
        name = folder + f'/cif10_tiny_pretrained_k_{n_loops}_run_{n_repeat}'

        training.train_and_test_rvit(model, train_loader, test_loader, optimizer, device, epochs=config['n_epochs'],
                                     save=True, name=name, criterion=criterion, wandb_log=True,
                                     scheduler=scheduler, n_loops=n_loops, train_option=config['train_option'])
        wandb.finish()
