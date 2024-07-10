import os
import sys
import torch
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import torch.nn as nn
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from pckgs.data import datasets
from config.definitions import NETWORKS_DIR
from pckgs.networks.utils.training import train_and_test_CNN
from pckgs.networks.PET import PETVGG, PETResNet, PETAlexNet


if __name__ == '__main__':
    params = {
        'AlexNet':
            {
                'lr': 0.0022,
                'lr_decay': 0.87,
                'optimizer': 'SGD',
                'model_architecture': PETAlexNet(my_pretrained_model=torchvision.models.alexnet(
                    weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)),
            },
        'ResNet':
            {
                'lr': 0.00444,
                'lr_decay': 0.72,
                'optimizer': 'SGD',
                'model_architecture': PETResNet(my_pretrained_model=torchvision.models.resnet18(
                    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)),
            },
        'VGG':
            {
                'lr': 0.0063,
                'lr_decay': 0.96,
                'optimizer': 'SGD',
                'model_architecture': PETVGG(my_pretrained_model=torchvision.models.vgg11_bn(
                    weights=torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1)),
            },
    }

    networks = ['AlexNet', 'ResNet', 'VGG']

    for current_network in networks:
        hp_defaults = dict(
            lr=params[current_network]['lr'],
            lr_decay=params[current_network]['lr_decay'],
            optimizer=params[current_network]['optimizer'],
            save=True,
            batch_size=32,
            n_epochs=1,     # testing only
            weight_decay=0.01,
            im_size=224
        )
        my_model = params[current_network]['model_architecture']

        cuda_idx = 0
        device = torch.device(f'cuda:{cuda_idx}' if torch.cuda.is_available() else "cpu")
        wandb.init(config=hp_defaults, project="RViT_Experiments")
        config = wandb.config

        folder = NETWORKS_DIR + '/PET/CNNs'
        model_path = folder + f'/pet_{current_network}_pretrained'
        my_model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        my_model.to(device)

        train_loader, test_loader, inp_shape, classes = datasets.load_data('PET',
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
                                        num_epochs=config['n_epochs'], train=False)

        wandb.finish()
