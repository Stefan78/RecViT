import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from pckgs.adversarials.utils import load_adversarial_examples
from pckgs.data.datasets import Transformation
from pckgs.networks.PET import PETVGG, PETResNet, PETAlexNet
from pckgs.networks.C10 import CIFAR10_AlexNet, CIFAR10_ResNet, CIFAR10_VGG11

from config.definitions import NETWORKS_DIR, AES_DIR, DATA_DIR


def _load_model_eval(network, dataset='PET'):
    if dataset == 'PET':
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

    elif dataset == 'CIFAR_10':
        folder = NETWORKS_DIR + '/CIFAR_10/CNNs'
        model_path = folder + f'/alex_net_finetune.pth'
        model = CIFAR10_AlexNet()
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    else:
        raise ValueError('Dataset not found')

    model.eval()

    return model, dataset


def softmax(x):
    x = x - np.max(x, axis=1)
    return np.exp(x)/np.sum(np.exp(x), axis=1)


def generate_ae_sample(network='AlexNet', name='', num_aes=2):
    """
    generates sample of AEs - 2x cif10, 2x PET with evaluation on AlexNet
    """
    # cif_AE_loader = load_adversarial_examples('CIFAR_10', batch_size=1, shuffle=True)

    classes_pet = {
        0: 'Abyssinian',
        1: 'American bulldog',
        2: 'American pit bull terrier',
        3: 'basset_hound',
        4: 'beagle',
        5: 'Bengal',
        6: 'Birman',
        7: 'Bombay',
        8: 'Boxer',
        9: 'British shorthair',
        10: 'Chihuahua',
        11: 'Egyptian Mau',
        12: 'English cocker spaniel',
        13: 'English setter',
        14: 'German shorthaired',
        15: 'Great pyrenees',
        16: 'Havanese',
        17: 'Japanese chin',
        18: 'Keeshond',
        19: 'Leonberger',
        20: 'Maine Coon',
        21: 'Miniature pinscher',
        22: 'Newfoundland',
        23: 'Persian',
        24: 'Pomeranian',
        25: 'Pug',
        26: 'Ragdoll',
        27: 'Russian Blue',
        28: 'Saint bernard',
        29: 'Samoyed',
        30: 'Scottish terrier',
        31: 'Shiba inu',
        32: 'Siamese',
        33: 'Sphynx',
        34: 'Staffordshire bull terrier',
        35: 'Wheaten terrier',
        36: 'Yorkshire terrier',
    }
    classes_cif10 = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }

    invTrans = Transformation('imagenet_statistics').inverse_transformation()
    imgs = []

    cif_ae_loader = load_adversarial_examples('CIFAR_10', 'low', batch_size=1, shuffle=True, loader='all')
    model, _ = _load_model_eval(network, dataset='CIFAR_10')

    for idx, (orig, adv, target) in enumerate(cif_ae_loader):
        if idx >= num_aes:
            break
        output_orig = model(orig.clone().detach()).detach().cpu().numpy()
        sft_out_orig = softmax(output_orig)

        output_adv = model(adv.clone().detach()).detach().cpu().numpy()
        sft_out_adv = softmax(output_adv)

        img_orig = np.moveaxis(invTrans(orig).squeeze().detach().cpu().numpy(), 0, 2)
        img_adv = np.moveaxis(invTrans(adv).squeeze().detach().cpu().numpy(), 0, 2)

        imgs.append([img_orig, np.max(sft_out_orig) * 100, classes_cif10[np.argmax(sft_out_orig)],
                     img_adv, np.max(sft_out_adv) * 100, classes_cif10[np.argmax(sft_out_adv)]])

    pet_ae_loader = load_adversarial_examples('PET', 'low', batch_size=1, shuffle=True, loader='all')
    model, _ = _load_model_eval(network)

    for idx, (orig, adv, target, seg_mask) in enumerate(pet_ae_loader):
        if idx > 300:
            break
        output_orig = model(orig.clone().detach()).detach().cpu().numpy()
        sft_out_orig = softmax(output_orig)

        output_adv = model(adv.clone().detach()).detach().cpu().numpy()
        sft_out_adv = softmax(output_adv)

        img_orig = np.moveaxis(invTrans(orig).squeeze().detach().cpu().numpy(), 0, 2)
        img_adv = np.moveaxis(invTrans(adv).squeeze().detach().cpu().numpy(), 0, 2)

        if len(classes_pet[np.argmax(sft_out_orig)]) > 9 or len(classes_pet[np.argmax(sft_out_adv)]) > 9 or (
                np.argmax(sft_out_orig) == np.argmax(sft_out_adv)):
            continue

        imgs.append([img_orig, np.max(sft_out_orig)*100, classes_pet[np.argmax(sft_out_orig)],
                     img_adv, np.max(sft_out_adv)*100, classes_pet[np.argmax(sft_out_adv)]])

    # PLOTTING
    rows, cols = 2, 2*num_aes
    fig, axs = plt.subplots(rows, cols, figsize=(10*num_aes, 13))
    for i in range(cols):
        axs[0, i].imshow(imgs[i][0])
        axs[0, i].axis('off')
        caption = f'{np.max(imgs[i][1]):2.0f}% ({imgs[i][2]})'
        axs[0, i].set_title(caption, fontsize=40)

        axs[1, i].imshow(imgs[i][3])
        axs[1, i].axis('off')
        caption = f'{np.max(imgs[i][4]):2.0f}% ({imgs[i][5]})'
        axs[1, i].set_title(caption, fontsize=40)

    plt.tight_layout()
    plt.plot()
    plt.savefig(name)


def show_pet_aes_with_mask(n=4):
    pet_AE_loader = load_adversarial_examples('PET', 'low', batch_size=1, shuffle=True, loader='all')
    invTrans = Transformation('imagenet_statistics').inverse_transformation()
    fig, axs = plt.subplots(3, n)
    for idx, (orig, adv, target, seg_mask) in enumerate(pet_AE_loader):
        if idx == n:
            break

        orig, adv = orig[0, :, :, :].numpy(), adv[0, :, :, :].numpy()
        target, seg_mask = target[0, :].numpy(), seg_mask[0, :, :, :].numpy()

        orig = np.array(invTrans(torch.tensor(orig)))
        adv = np.array(invTrans(torch.tensor(adv)))

        # orig = np.clip(orig, 0, 1)
        # adv = np.clip(adv, 0, 1)

        img = np.moveaxis(orig, 0, 2)
        axs[0][idx].imshow(img)
        axs[0][idx].axis('off')

        adv = np.moveaxis(adv, 0, 2)
        axs[1][idx].imshow(adv)
        axs[1][idx].axis('off')

        seg_mask = np.moveaxis(seg_mask, 0, 2)
        axs[2][idx].imshow(seg_mask)
        axs[2][idx].axis('off')

    plt.show()


def generate_random_image(shape=(100, 100, 3)):
    """Generate a random image."""
    return np.random.rand(*shape)


def plot_images_with_captions(rows, cols):
    """
    Plot randomly generated images with captions in a grid.

    Parameters:
    - rows: Number of rows in the grid.
    - cols: Number of columns in the grid.
    """
    images = [generate_random_image() for _ in range(rows * cols)]
    captions = [f'Caption {i+1}' for i in range(rows * cols)]

    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            axs[i, j].imshow(images[index])
            axs[i, j].axis('off')
            axs[i, j].set_title(captions[index], fontsize=20)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_images_with_captions(2, 2)
    for i in range(50):
        generate_ae_sample(name=f'example_{i}', num_aes=3)
