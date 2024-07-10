import os
import sys
import numpy as np
import pickle
import torchvision
import torch
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from pckgs.networks.PET import PETVGG, PETResNet, PETAlexNet
from config.definitions import NETWORKS_DIR, AES_DIR, DATA_DIR
from pckgs.data.datasets import Transformation


def load_adversarials(network, eps, shuffle=False, must_fool_all=True, use_batching=True, batch_size=10,
                      load_subset=False, n_adversarials=1000):
    original, x_test_adv, y_test_adv, y_test_seg = [], [], [], []
    dataset = 'PET'

    # Load all AEs given a list of eps
    for ind, e in tqdm.tqdm(enumerate(eps)):
        with open(AES_DIR + f'/{dataset}/all/PGD_{network}_eps_{e:2.4f}.pkl', 'rb') as f:
            [o, x, y, m] = pickle.load(f)
        if ind == 0:
            original, x_test_adv, y_test_adv, y_test_seg = o, x, y, m
        else:
            original = np.concatenate((original, o))
            x_test_adv = np.concatenate((x_test_adv, x))
            y_test_adv = np.concatenate((y_test_adv, y))
            y_test_seg = np.concatenate((y_test_seg, m))

    if load_subset and x_test_adv.shape[0] >= n_adversarials:
        print(f'Subset of {n_adversarials} adversarial examples selected!')
        random_indices = np.arange(x_test_adv.shape[0])
        random_indices = np.random.permutation(random_indices)
        random_indices = random_indices[:n_adversarials]

        original = original[random_indices, :, :, :]
        x_test_adv = x_test_adv[random_indices, :, :, :]
        y_test_adv = y_test_adv[random_indices, :]
        y_test_seg = y_test_seg[random_indices, :, :, :]

    # Transform to match imagenet statistics due to ART - where the transform had to be Identity
    trans = Transformation('imagenet_statistics').transformation()
    original = trans(torch.Tensor(original)).numpy()
    x_test_adv = trans(torch.Tensor(x_test_adv)).numpy()

    # Test, whether the loaded adversarial examples really fool the network
    if must_fool_all:
        print(f'Initial number of AEs: {original.shape[0]}')
        for network in ['AlexNet', 'ResNet', 'VGG']:
            model, _ = _load_model_eval(network)
            print(f'Model {network} loaded')

            if original.shape[0] > batch_size and use_batching:
                orig_tmp = np.array_split(original, original.shape[0]//batch_size, axis=0)
                adv_tmp = np.array_split(x_test_adv, original.shape[0]//batch_size, axis=0)
                targets = np.array_split(y_test_adv, original.shape[0]//batch_size, axis=0)
                seg_masks = np.array_split(y_test_seg, original.shape[0]//batch_size, axis=0)

                corr_msk = []

                for o, a, t, m in zip(orig_tmp, adv_tmp, targets, seg_masks):
                    output = model(torch.tensor(a)).detach().cpu().numpy()
                    corr_msk.append(np.argmax(output, axis=1) != np.argmax(t, axis=1))

                corr_msk = np.concatenate(corr_msk).ravel()

            else:
                output = model(torch.tensor(x_test_adv)).detach().cpu().numpy()
                corr_msk = np.argmax(output, axis=1) != np.argmax(y_test_adv, axis=1)

            original = original[corr_msk, :, :, :]
            x_test_adv = x_test_adv[corr_msk, :, :, :]
            y_test_adv = y_test_adv[corr_msk, :]
            y_test_seg = y_test_seg[corr_msk, :, :, :]

            print(f'Number of AEs after testing on {network}: {original.shape[0]}')

    # randomize order
    if shuffle:
        perm = np.random.permutation(original.shape[0])
        original = original[perm, :, :, :]
        x_test_adv = x_test_adv[perm, :, :, :]
        y_test_adv = y_test_adv[perm, :]
        y_test_seg = y_test_seg[perm, :, :, :]

        print('Adversarial examples shuffled.')

    print(f'Successfully loaded {original.shape[0]} adversarial examples.')
    return original, x_test_adv, y_test_adv, y_test_seg


def _load_model_eval(network, dataset='PET'):
    if network == 'AlexNet':
        model = PETAlexNet(my_pretrained_model=torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1))
    elif network == 'ResNet':
        model = PETResNet(my_pretrained_model=torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1))
    elif network == 'VGG':
        model = PETVGG(my_pretrained_model=torchvision.models.vgg11_bn(
            weights=torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1))
    else:
        raise ValueError('Please select valid trained model!')

    pretrained_model = NETWORKS_DIR+f'/{dataset}/CNNs/pet_{network}_pretrained'
    model.load_state_dict(torch.load(pretrained_model, map_location='cuda:0'))
    model.eval()

    return model, dataset


def cross_validate_aes_and_save(pert_intensity, load_subset=False, n_adversarials=1000):
    # perturbations = list(np.arange(0.007, 0.601, 0.015))  # the full range of perturbations

    if pert_intensity == 'low':
        perturbations = list(np.arange(0.007, 0.217, 0.015))    # small perturbations
    elif pert_intensity == 'medium':
        perturbations = list(np.arange(0.217, 0.412, 0.015))  # medium perturbations
    elif pert_intensity == 'high':
        perturbations = list(np.arange(0.412, 0.601, 0.015))  # high perturbations
    else:
        raise ValueError(f'Perturbation intensity {pert_intensity} not defined!')

    original_a, x_test_adv_a, y_test_adv_a, y_test_seg_a = load_adversarials('AlexNet', eps=perturbations,
                                                                             must_fool_all=True, use_batching=True,
                                                                             load_subset=load_subset,
                                                                             n_adversarials=n_adversarials)
    original_r, x_test_adv_r, y_test_adv_r, y_test_seg_r = load_adversarials('ResNet', eps=perturbations,
                                                                             must_fool_all=True, use_batching=True,
                                                                             load_subset=load_subset,
                                                                             n_adversarials=n_adversarials)
    original_v, x_test_adv_v, y_test_adv_v, y_test_seg_v = load_adversarials('VGG', eps=perturbations,
                                                                             must_fool_all=True, use_batching=True,
                                                                             load_subset=load_subset,
                                                                             n_adversarials=n_adversarials)

    original = np.concatenate((original_a, original_r, original_v))
    x_test_adv = np.concatenate((x_test_adv_a, x_test_adv_r, x_test_adv_v))
    y_test_adv = np.concatenate((y_test_adv_a, y_test_adv_r, y_test_adv_v))
    y_test_seg = np.concatenate((y_test_seg_a, y_test_seg_r, y_test_seg_v))

    del original_a, x_test_adv_a, y_test_adv_a, y_test_seg_a
    del original_r, x_test_adv_r, y_test_adv_r, y_test_seg_r
    del original_v, x_test_adv_v, y_test_adv_v, y_test_seg_v

    filename = DATA_DIR + f'/aes/PET/adv_PGD_pet_{pert_intensity}.pkl'

    with open(filename, 'wb') as f:
        pickle.dump([original, x_test_adv, y_test_adv, y_test_seg], f)


def merge_aes(network):
    perturbations = list(np.arange(0.007, 0.217, 0.015))

    original, x_test_adv, y_test_adv, y_test_seg = load_adversarials(network, eps=perturbations, must_fool_all=True,
                                                                     use_batching=True, load_subset=False,)

    filename = DATA_DIR + f'/aes/PET/adv_PGD_pet_{network}.pkl'

    with open(filename, 'wb') as f:
        pickle.dump([original, x_test_adv, y_test_adv, y_test_seg], f)


if __name__ == '__main__':  # make sure AEs are saved in "all" folder
    cross_validate_aes_and_save('low', load_subset=False)
    merge_aes('AlexNet')
    merge_aes('ResNet')
    merge_aes('VGG')
