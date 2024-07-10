"""
This file generates and saves attention maps for PET images (original vs adversarial)
Also, it calculates and saves the similarity score of the overlaps
"""

import os
import sys
import torch
import tqdm

from experiments_utils import HeatMap
import numpy as np
import matplotlib.pyplot as plt
from network_types import type_pet_m1_baseline, type_pet_m2_baseline, type_pet_m1_all, type_pet_m2_all

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from pckgs.networks.utils.vits import get_rvit_activations
from pckgs.networks.network_utils import load_trained_network
from pckgs.data.datasets import Transformation
from pckgs.adversarials.utils import load_adversarial_examples
import pickle


if __name__ == '__main__':
    # setting the output directory
    save_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    save_dir = os.path.abspath(os.path.join(save_dir, os.pardir))
    save_dir = os.path.join(save_dir, 'results', 'experiment_pet')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cuda_idx = 0
    batch_size = 1  # do not change (due to overlap computation)
    device = f'cuda:{cuda_idx}' if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------------------------------------------------------
    # fixme: define your options
    nets = type_pet_m2_all
    n_loops = 2     # number of loops the network was trained on
    run_no = 0      # usually from <0, 4>
    count = 100    # number of processed images

    # ------------------------------------------------------------------------------------------------------------------
    # Load the model and define its parameters
    # Load the training/testing data - only the testing data are relevant

    model_params = {
        'extra_tiny': (48, 6, 72, 6, 6),
        'tiny': (224, 8, 192, 12, 3),
    }

    im_size, patch_size, embed_dim, depth, num_heads = model_params[nets['model_name']]
    ds = 'CIFAR_10' if nets['dataset'] == 'CIFAR10' else 'PET'

    model = load_trained_network(name=nets['model_name'], dataset=ds, n_loops=n_loops,
                                 pretrained=nets['pretrained'], device=device, run_no=run_no,
                                 on_off=nets['on_off'], reg_1000=nets['reg_1000'],
                                 method2=nets['method2'], tiny_patch=patch_size, suffix='')  # fixme manually change the suffix

    # _, test_loader, inp_shape, classes = datasets.load_data(dataset=nets['dataset'], batch_size_train=batch_size,
    #                                                         batch_size_test=batch_size, imsize=im_size,
    #                                                         get_seg_mask=True)

    # ------------------------------------------------------------------------------------------------------------------
    invTrans = Transformation('imagenet_statistics').inverse_transformation()
    Trans = Transformation('imagenet_statistics').transformation()

    # ------------------------------------------------------------------------------------------------------------------

    scores = np.zeros((count, 12 * n_loops))
    scores_adv = np.zeros((count, 12 * n_loops))
    tok_no = n_loops  # the loop number from which the activations are extracted

    torch.manual_seed(49)
    adv_loader = load_adversarial_examples('PET', 'low', batch_size=1, shuffle=True, loader='all')
    adv_loader = iter(adv_loader)

    this_many = 0

    for adv_orig_pair_idx in tqdm.tqdm(range(count)):
        orig, adv, target, seg_mask = next(adv_loader)

        adv = adv.to(device)
        orig = orig.to(device)

        orig_act, orig_tok, _ = get_rvit_activations(model, orig, batch_size=batch_size, repeats=tok_no,
                                                     layer=list(range(depth)), dim=embed_dim, num_heads=num_heads,
                                                     blur='no_blur')  # fixme manually set blur ('no_blur', 'blur', 'inv_blur')

        adv_act, adv_tok, _ = get_rvit_activations(model, adv, batch_size=batch_size, repeats=tok_no,
                                                   layer=list(range(depth)), dim=embed_dim, num_heads=num_heads,
                                                   blur='no_blur')  # fixme manually set blur ('no_blur', 'blur', 'inv_blur')

        adv = invTrans(adv)
        orig = invTrans(orig)

        for lay in list(range(tok_no * 12)):
            orig_att_data = orig_act[lay]  # the index determines the layer

            heat_map = HeatMap(orig_att_data, patch_size=patch_size, input_image=orig, segmentation_mask=seg_mask)
            heat_map.visualize_map(visualize_token=True, merge_heads=True, blur_factor=7, scale_overlay=25, save=True,
                                   save_to=save_dir,
                                   save_name=f'idx_{adv_orig_pair_idx}_layer{lay}_orig')
            scores[adv_orig_pair_idx][lay] = heat_map.get_overlap_score()

        for lay in list(range(tok_no * 12)):
            adv_att_data = adv_act[lay]  # the index determines the layer

            heat_map = HeatMap(adv_att_data, patch_size=patch_size, input_image=adv, segmentation_mask=seg_mask)
            heat_map.visualize_map(visualize_token=True, merge_heads=True, blur_factor=7, scale_overlay=25,
                                   save=True, save_to=save_dir,
                                   save_name=f'idx_{adv_orig_pair_idx}_layer{lay}_adv')
            scores_adv[adv_orig_pair_idx][lay] = heat_map.get_overlap_score()

        del orig_act, orig_tok, adv_act, adv_tok, adv, orig, seg_mask, orig_att_data, adv_att_data

    x = np.arange(12*n_loops)

    y_1 = np.mean(scores, axis=0)
    err_1 = np.std(scores, axis=0)

    y_2 = np.mean(scores_adv, axis=0)
    err_2 = np.std(scores_adv, axis=0)

    fig, ax = plt.subplots(figsize=(10, 3))

    plt.errorbar(x, y_1, yerr=err_1, fmt='-o', capsize=5, c="cornflowerblue", label='Original')
    plt.errorbar(x, y_2, yerr=err_2, fmt='-o', capsize=5, c="coral", label='Adversarial')

    ax.legend(prop={'size': 19})
    ax.get_xaxis().set_ticks([])
    ax.set_ylim([-0.02, 0.06])    # fixme set the y_limit according to the needs

    plt.ylabel('Cosine sim.', fontsize=22)
    plt.xlabel('Layers', fontsize=22)
    plt.yticks(fontsize=16)
    plt.savefig('cos_similarity')

    plt.grid(True, linestyle='dashed')
    plt.show()

    with open('similarities', 'wb') as f:
        pickle.dump([x, y_1, y_2, err_1, err_2], f)
