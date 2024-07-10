import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from pckgs.networks.utils import training
from pckgs.networks.network_utils import load_trained_network
from network_types import type_pet_m2_all
from pckgs.adversarials.utils import load_adversarial_examples
from pckgs.data import datasets
import numpy as np


if __name__ == '__main__':
    nets = type_pet_m2_all  # fixme define the network type

    cuda_idx = 0
    batch_size = 8
    device = f'cuda:{cuda_idx}' if torch.cuda.is_available() else "cpu"

    n_loops = 3  # number of loops the network was trained on

    # ------------------------------------------------------------------------------------------------------------------
    # Load the model and define its parameters
    # Load the training/testing data - only the testing data are relevant

    model_params = {
        'tiny': (224, 8, 192, 12, 3),
    }

    im_size, patch_size, embed_dim, depth, num_heads = model_params[nets['model_name']]

    ds = 'CIFAR_10' if nets['dataset'] == 'CIFAR10' else 'PET'

    acc_stats = []
    for run_no in range(5):
        model = load_trained_network(name=nets['model_name'], dataset=ds, n_loops=n_loops,
                                     pretrained=nets['pretrained'], device=device, run_no=run_no,
                                     on_off=nets['on_off'], reg_1000=nets['reg_1000'],
                                     method2=nets['method2'], tiny_patch=patch_size, suffix='_blurinv')  # fixme change the suffix according to the investigated network
        _, test_loader, _, _ = datasets.load_data('PET', batch_size_train=32, batch_size_test=90,
                                                  transform="imagenet_statistics", imsize=224)
        print(f'Accuracy of RViT (loops={n_loops}) PET on the test set:')
        acc = training.test_rvit(model, test_loader, n_loops, device,
                                 random_transform=False, blur=True, inv_blur=True)  # fixme set random_transform, blur, and inv_blur according to the needs
        print(acc)
        acc_stats.append(acc[-1])

    acc_test = np.array(acc_stats)

    print(' ------------------------------------- ')
    print(f'Average accuracy on the TEST SET: {np.mean(acc_stats):2.4f}')
    print(f'Standard deviation on the TEST SET: {np.std(acc_stats):2.4f}')
    print(' ------------------------------------- ')

    for perturbation in ['low']:        # ['low', 'AlexNet', 'ResNet', 'VGG']
        acc_stats = []
        pet_adv_loader = load_adversarial_examples('PET', perturbation, batch_size=90, shuffle=True,
                                                   loader='adversarial')
        for run_no in range(5):
            model = load_trained_network(name=nets['model_name'], dataset=ds, n_loops=n_loops,
                                         pretrained=nets['pretrained'], device=device, run_no=run_no,
                                         on_off=nets['on_off'], reg_1000=nets['reg_1000'],
                                         method2=nets['method2'], tiny_patch=patch_size, suffix='_blurinv')  # fixme change the suffix according to the investigated network

            print(f'Accuracy of RViT (loops={n_loops}) PET on {perturbation} adversarials:')
            acc = training.test_rvit(model, pet_adv_loader, n_loops, device,
                                     random_transform=False, blur=True, inv_blur=True)  # fixme set random_transform, blur, and inv_blur according to the needs
            print(acc)
            acc_stats.append(acc[-1])

        acc_adv = np.array(acc_stats)

        print(' ------------------------------------- ')
        print(f'Average accuracy on {perturbation}: {np.mean(acc_stats):2.4f}')
        print(f'Standard deviation on {perturbation}: {np.std(acc_stats):2.4f}')
        print(' ------------------------------------- ')

    print('\n Acc test:', acc_test)
    print('Acc adv:', acc_adv)
    print(f'Correlation Acc {np.corrcoef(acc_test, acc_adv)}')
