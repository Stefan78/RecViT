import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from pckgs.networks.utils import training
from pckgs.networks.network_utils import load_trained_network

from network_types import type_2
from pckgs.adversarials.utils import load_adversarial_examples
from pckgs.data import datasets
import numpy as np


if __name__ == '__main__':
    nets = type_2

    cuda_idx = 0
    batch_size = 8
    device = f'cuda:{cuda_idx}' if torch.cuda.is_available() else "cpu"

    n_loops = 1  # number of loops the network was trained on
    tok_no = n_loops  # the loop number from which the activations are extracted

    # ------------------------------------------------------------------------------------------------------------------
    # Load the model and define its parameters
    # Load the training/testing data - only the testing data are relevant

    model_params = {
        'extra_tiny': (48, 6, 72, 6, 6),
        'tiny': (224, 16, 192, 12, 3),
    }

    im_size, patch_size, embed_dim, depth, num_heads = model_params[nets['model_name']]
    ds = 'CIFAR_10' if nets['dataset'] == 'CIFAR10' else 'PET'

    acc_stats = []
    for run_no in range(5):
        model = load_trained_network(name=nets['model_name'], dataset=ds, n_loops=n_loops,
                                     pretrained=nets['pretrained'], device=device, run_no=run_no,
                                     on_off=nets['on_off'], reg_1000=nets['reg_1000'],
                                     method2=nets['method2'], tiny_patch=patch_size, suffix='_random_blur_1')    # fixme change the suffix according to the investigated network

        _, test_loader, _, _ = datasets.load_data('CIFAR10', batch_size_train=32, batch_size_test=500,
                                                  transform="special_2", imsize=224)
        print(f'Accuracy of RViT (loops={n_loops}) PET on the test set:')
        acc = training.test_rvit(model, test_loader, n_loops, device,
                                 random_transform=False, blur=False, inv_blur=False,
                                 random_blur=1, blur_value=(35, 12))  # random_blur=n_blurs, blur_value=(35, 12)
        # fixme set random_transform, blur, and inv_blur according to the needs

        print(acc)
        acc_stats.append(acc[-1])
    acc_test = np.array(acc_stats)

    print(' ------------------------------------- ')
    print(f'Average accuracy on the TEST SET: {np.mean(acc_stats):2.4f}')
    print(f'Standard deviation on the TEST SET: {np.std(acc_stats):2.4f}')
    print(' ------------------------------------- ')

    for perturbation in ['low']:
        acc_stats = []
        pet_adv_loader = load_adversarial_examples('CIFAR_10', perturbation, batch_size=500, shuffle=True,
                                                   loader='adversarial')
        for run_no in range(5):
            model = load_trained_network(name=nets['model_name'], dataset=ds, n_loops=n_loops,
                                         pretrained=nets['pretrained'], device=device, run_no=run_no,
                                         on_off=nets['on_off'], reg_1000=nets['reg_1000'],
                                         method2=nets['method2'], tiny_patch=patch_size, suffix='_random_blur_1')  # fixme change the suffix according to the investigated network

            print(f'Accuracy of RViT (loops={n_loops}) PET on {perturbation} adversarials:')
            acc = training.test_rvit(model, pet_adv_loader, n_loops, device,
                                     random_transform=False, blur=False, inv_blur=False,
                                     random_blur=1, blur_value=(35, 12))    # random_blur=n_blurs, blur_value=(35, 12)
            # fixme set random_transform, blur, and inv_blur according to the needs
            print(acc)
            acc_stats.append(acc[-1])

        acc_adv = np.array(acc_stats)

        print(' ------------------------------------- ')
        print(f'Average accuracy on {perturbation}: {np.mean(acc_stats):2.4f}')
        print(f'Standard deviation on {perturbation}: {np.std(acc_stats):2.4f}')
        print(' ------------------------------------- ')

    print('')
    print('Acc test:', acc_test)
    print('Acc adv:', acc_adv)
    print('')
    print(f'Correlation Acc {np.corrcoef(acc_test, acc_adv)}')
    print('End')
