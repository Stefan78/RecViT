# this file is used when searching for best hyperparams for ResNet

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from pckgs.data import datasets
import wandb
from config.definitions import NETWORKS_DIR
from pckgs.networks.utils.training import train_and_test_CNN
from pckgs.networks.PET import PETVGG, PETResNet, PETAlexNet


if __name__ == '__main__':
    pretrained_model = torchvision.models.resnet18(weights=True)
    hp_defaults = dict(
        lr=0.00444,
        lr_decay=0.72,
        optimizer='SGD',
        batch_size=64,
        n_epochs=30,
        weight_decay=0.01,
        im_size=224
    )

    cuda_idx = 0
    device = torch.device(f'cuda:{cuda_idx}' if torch.cuda.is_available() else "cpu")

    wandb.init(config=hp_defaults, project="RViT_Experiments")
    config = wandb.config
    my_model = PETResNet(my_pretrained_model=pretrained_model)

    train_loader, test_loader, inp_shape, classes = datasets.load_data('PET',
                                                                       transform='imagenet_statistics',
                                                                       batch_size_train=config['batch_size'],
                                                                       batch_size_test=config['batch_size'],
                                                                       imsize=config['im_size'])

    dataset_sizes = {'train': 3680, 'test': 3669}

    my_model = my_model.to(device)
    criterion = nn.CrossEntropyLoss()

    if config['optimizer'] == 'SGD':
        optimizer_conv = optim.SGD(my_model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['optimizer'] == 'Adam':
        optimizer_conv = optim.Adam(my_model.parameters(), lr=config['lr'])

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=config['lr_decay'])

    model_conv = train_and_test_CNN(my_model, criterion, optimizer_conv, exp_lr_scheduler, device=device,
                                    dataset_sizes=dataset_sizes, test_loader=test_loader, train_loader=train_loader,
                                    num_epochs=config['n_epochs'], train=True)

    save = False
    if save:
        folder = NETWORKS_DIR + '/PET/CNNs'
        if not os.path.exists(folder):
            os.makedirs(folder)
        PATH = folder + f'/pet_ResNet_pretrained'

        torch.save(my_model.state_dict(), PATH)
