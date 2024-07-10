import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torchvision
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from pckgs.adversarials.utils import compute_attack
from pckgs.networks.PET import PETVGG, PETResNet, PETAlexNet
from config.definitions import DATA_DIR
from config.definitions import NETWORKS_DIR


if __name__ == '__main__':
    # Generate adversarial examples on PET dataset given their segmentation mask.
    # Adversarial examples for CIFAR-10 computed outside of this project!

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-nts', '--networks', help='Network index/indices', required=True)
    args = vars(parser.parse_args())

    networks = int(args['networks'])
    cuda_idx = 0    # does not work with cuda_idx=1 due to ART compatibility

    dataset = 'PET'
    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
    attack = 'L_inf'

    if networks == 1:
        types = ['VGG']
    elif networks == 2:
        types = ['AlexNet', 'ResNet']
    elif networks == 3:
        types = ['AlexNet', 'ResNet', 'VGG']
    else:
        raise ValueError('Network value not defined!')

    for typ in types:
        if typ == 'AlexNet':
            model = PETAlexNet(my_pretrained_model=torchvision.models.alexnet(
                weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1))
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # default, (not used)
        elif typ == 'ResNet':
            model = PETResNet(my_pretrained_model=torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1))
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # default, (not used)
        elif typ == 'VGG':
            model = PETVGG(my_pretrained_model=torchvision.models.vgg11_bn(
                weights=torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1))
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # default, (not used)
        else:
            raise ValueError('Please select valid trained model!')

        pretrained_model = NETWORKS_DIR + f'/{dataset}/CNNs/pet_' + typ + '_pretrained'
        model.to(device)

        # L_inf (PGD) attack for different perturbation magnitudes
        random_in = 5

        eps = np.arange(0.007, 0.601, 0.015)

        for e in eps:
            print(f'Generating AEs for e={e:2.4f}')

            folder = DATA_DIR + f'/aes/{dataset}'
            if not os.path.exists(folder):
                os.makedirs(folder)

            compute_attack(dataset, model, pretrained_model, 'Linf', optimizer,
                           folder + f'/PGD_{typ}_eps_{e:2.4f}.pkl', device, imsize=224,
                           exclude_incorrect=True, eps=e, eps_step=0.001, max_iter=250, num_rand_init=random_in,
                           num_inputs=192, batch=64, transform='identity')
