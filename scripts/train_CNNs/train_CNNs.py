import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
import torch
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from pckgs.data import datasets
import wandb
from config.definitions import NETWORKS_DIR
from pckgs.networks.utils.training import train_and_test_CNN
from pckgs.networks.PET import PETVGG, PETResNet, PETAlexNet
import torch.nn as nn


if __name__ == '__main__':
    params = {
        'AlexNet':
            {
                'lr': 0.0022,
                'lr_decay': 0.87,
                'optimizer': 'SGD',
                'model_architecture': PETAlexNet(my_pretrained_model=torchvision.models.alexnet(weights=True)),
            },
        'ResNet':
            {
                'lr': 0.00444,
                'lr_decay': 0.72,
                'optimizer': 'SGD',
                'model_architecture': PETResNet(my_pretrained_model=torchvision.models.resnet18(weights=True)),
            },
        'VGG':
            {
                'lr': 0.0063,
                'lr_decay': 0.96,
                'optimizer': 'SGD',
                'model_architecture': PETVGG(my_pretrained_model=torchvision.models.vgg11_bn(weights=True)),
            },
    }

    networks = ['AlexNet', 'ResNet', 'VGG']
    for current_network in networks:
        hp_defaults = dict(
            lr=params[current_network]['lr'],
            lr_decay=params[current_network]['lr_decay'],
            optimizer=params[current_network]['optimizer'],
            save=True,
            batch_size=64,
            n_epochs=30,
            weight_decay=0.01,
            im_size=224,
            im_crop=300
        )
        my_model = params[current_network]['model_architecture']

        cuda_idx = 0
        device = torch.device(f'cuda:{cuda_idx}' if torch.cuda.is_available() else "cpu")
        wandb.init(config=hp_defaults, project="RViT_Experiments")
        config = wandb.config

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
        else:
            raise ValueError(f'Invalid optimizer value: {config["optimizer"]}')

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=config['lr_decay'])

        model_conv = train_and_test_CNN(my_model, criterion, optimizer_conv, exp_lr_scheduler, device=device,
                                        dataset_sizes=dataset_sizes, test_loader=test_loader, train_loader=train_loader,
                                        num_epochs=config['n_epochs'], train=True)

        if config['save']:
            folder = NETWORKS_DIR + '/PET/CNNs'
            if not os.path.exists(folder):
                os.makedirs(folder)
            PATH = folder + f'/pet_{current_network}_pretrained'

            torch.save(my_model.state_dict(), PATH)
        wandb.finish()
