"""
Generates and saves attention maps for CIFAR-10 images (original vs adversarial)
(Works for the network without suffix i.e., no invblur, blur, ...)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
import torch
import tqdm
from pckgs.data import datasets
from pckgs.networks.utils.vits import get_rvit_activations
from network_types import *
from experiments_utils import HeatMap
from pckgs.networks.network_utils import load_trained_network
from pckgs.data.datasets import Transformation
from pckgs.adversarials.utils import load_adversarial_examples


if __name__ == '__main__':
    # setting the output directory
    save_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    save_dir = os.path.abspath(os.path.join(save_dir, os.pardir))
    save_dir = os.path.join(save_dir, 'results', 'experiment_cifar10')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cuda_idx = 0
    batch_size = 1
    device = f'cuda:{cuda_idx}' if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------------------------------------------------------
    # fixme: define your options
    nets = type_2  # network type (int: <1, 9>)
    net_type = nets['idx']  # index number, should correspond to the network above

    count = 5
    n_loops = 1  # number of loops the network was trained on
    run_no = 0  # usually from <0, 4>

    # ------------------------------------------------------------------------------------------------------------------
    # Load the model and define its parameters
    # Load the training/testing data - only the testing data are relevant

    model_params = {
        'extra_tiny': (48, 6, 72, 6, 6),
        'tiny': (224, 16, 192, 12, 3)
    }

    im_size, patch_size, embed_dim, depth, num_heads = model_params[nets['model_name']]
    ds = 'CIFAR_10' if nets['dataset'] == 'CIFAR10' else 'PET'

    model = load_trained_network(name=nets['model_name'], dataset=ds, n_loops=n_loops,
                                 pretrained=nets['pretrained'], device=device, run_no=run_no,
                                 on_off=nets['on_off'], reg_1000=nets['reg_1000'],
                                 method2=nets['method2'])

    train_loader, test_loader, inp_shape, classes = datasets.load_data(dataset=nets['dataset'],
                                                                       batch_size_train=batch_size,
                                                                       batch_size_test=batch_size,
                                                                       transform="special_2",
                                                                       imsize=im_size)

    adv_loader = load_adversarial_examples(dataset='CIFAR_10', batch_size=1, shuffle=True, loader='all')
    adv_loader = iter(adv_loader)

    # ------------------------------------------------------------------------------------------------------------------
    invTrans = Transformation('imagenet_statistics').inverse_transformation()
    Trans = Transformation('imagenet_statistics').transformation()

    # ------------------------------------------------------------------------------------------------------------------

    tok_no = n_loops  # the loop number up to which the activations are extracted

    for adv_orig_pair_idx in tqdm.tqdm(range(count)):
        orig_data, adv_data, target = next(adv_loader)

        adv_data = adv_data.to(device)
        orig_data = orig_data.to(device)

        adv_act, adv_tok, _ = get_rvit_activations(model, adv_data, batch_size=batch_size, repeats=tok_no,
                                                   layer=list(range(depth)), dim=embed_dim, num_heads=num_heads)
        orig_act, orig_tok, _ = get_rvit_activations(model, orig_data, batch_size=batch_size, repeats=tok_no,
                                                     layer=list(range(depth)), dim=embed_dim, num_heads=num_heads)

        adv_data = invTrans(adv_data)
        orig_data = invTrans(orig_data)

        for lay in list(range(tok_no*12)):
            adv_att_data = adv_act[lay]  # the index determines the layer

            heat_map = HeatMap(adv_att_data, patch_size=patch_size, input_image=orig_data, segmentation_mask=False)
            heat_map.visualize_map(visualize_token=True, merge_heads=True, blur_factor=7, scale_overlay=25, save=True,
                                   save_to=save_dir, save_name=f'idx_{adv_orig_pair_idx}_layer{lay}_adv')

        for lay in list(range(tok_no*12)):
            orig_att_data = orig_act[lay]  # the index determines the layer

            heat_map = HeatMap(orig_att_data, patch_size=patch_size, input_image=orig_data, segmentation_mask=False)
            heat_map.visualize_map(visualize_token=True, merge_heads=True, blur_factor=7, scale_overlay=25, save=True,
                                   save_to=save_dir, save_name=f'idx_{adv_orig_pair_idx}_layer{lay}_orig')
