import os
import sys
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from pckgs.data import datasets
from config.definitions import NETWORKS_DIR
from pckgs.networks.utils.training import train_and_test_CNN
from pckgs.networks.C10 import CIFAR10_AlexNet, CIFAR10_ResNet, CIFAR10_VGG11


if __name__ == '__main__':
    networks = ['alex_net', 'res_net', 'VGG_net']
    architectures = [CIFAR10_AlexNet(), CIFAR10_ResNet(), CIFAR10_VGG11()]

    for current_network, my_model in zip(networks, architectures):
        device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else "cpu")

        folder = NETWORKS_DIR + '/CIFAR_10/CNNs'
        model_path = folder + f'/{current_network}_finetune.pth'
        my_model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        my_model.to(device)

        train_loader, test_loader, inp_shape, classes = datasets.load_data('CIFAR10',
                                                                           batch_size_train=20,
                                                                           batch_size_test=20,
                                                                           imsize=224,
                                                                           transform='special_2')

        dataset_sizes = {'train': 50000, 'test': 10000}
        my_model = my_model.to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer_conv = optim.SGD(my_model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=1, gamma=0.9)

        model_conv = train_and_test_CNN(my_model, criterion, optimizer_conv, exp_lr_scheduler, device=device,
                                        dataset_sizes=dataset_sizes, test_loader=test_loader, train_loader=train_loader,
                                        num_epochs=1, train=False)
