from pckgs.data.datasets import Transformation
from pckgs.adversarials.utils import load_adversarial_examples
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms.functional as F


adv_loader = load_adversarial_examples('PET', 'low', batch_size=1, shuffle=True, loader='original')
invTrans = Transformation('imagenet_statistics').inverse_transformation()

for img, t in adv_loader:
    img = invTrans(img)

    fig, axs = plt.subplots(3, 4)

    # ----------------------------------------------------------------------------------------------------------------------
    # Adding the second and third row - blurred / inv blurred
    blr = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=4)

    img1 = np.moveaxis(img.squeeze().squeeze().detach().cpu().numpy(), 0, 2)
    axs[1, 0].imshow(img1)
    axs[1, 0].axis('off')

    axs[2, 3].imshow(img1)
    axs[2, 3].axis('off')

    img_copy = blr(img)
    img1 = np.moveaxis(img_copy.squeeze().squeeze().detach().cpu().numpy(), 0, 2)
    axs[1, 1].imshow(img1)
    axs[1, 1].axis('off')

    axs[2, 2].imshow(img1)
    axs[2, 2].axis('off')

    img_copy = blr(img_copy)
    img1 = np.moveaxis(img_copy.squeeze().squeeze().detach().cpu().numpy(), 0, 2)
    axs[1, 2].imshow(img1)
    axs[1, 2].axis('off')

    axs[2, 1].imshow(img1)
    axs[2, 1].axis('off')

    img_copy = blr(img_copy)
    img1 = np.moveaxis(img_copy.squeeze().squeeze().detach().cpu().numpy(), 0, 2)
    axs[1, 3].imshow(img1)
    axs[1, 3].axis('off')

    axs[2, 0].imshow(img1)
    axs[2, 0].axis('off')
    # ----------------------------------------------------------------------------------------------------------------------

    rot = np.random.rand() * 20 - 10
    trans = (np.random.randint(-5, 6), np.random.randint(-5, 6))
    scale = np.random.rand() * 0.2 + 0.9

    img1 = np.moveaxis(img.squeeze().squeeze().detach().cpu().numpy(), 0, 2)
    axs[0, 0].imshow(img1)
    axs[0, 0].axis('off')
    img = F.affine(img, rot, trans, scale, 0)

    img1 = np.moveaxis(img.squeeze().squeeze().detach().cpu().numpy(), 0, 2)
    axs[0, 1].imshow(img1)
    axs[0, 1].axis('off')
    img = F.affine(img, rot, trans, scale, 0)

    img1 = np.moveaxis(img.squeeze().squeeze().detach().cpu().numpy(), 0, 2)
    axs[0, 2].imshow(img1)
    axs[0, 2].axis('off')
    img = F.affine(img, rot, trans, scale, 0)

    img1 = np.moveaxis(img.squeeze().squeeze().detach().cpu().numpy(), 0, 2)
    axs[0, 3].imshow(img1)
    axs[0, 3].axis('off')

    axs[0, 0].text(-0.2, 0.5, 'RandT', size=25, ha="center", va="center", transform=axs[0, 0].transAxes,
                   rotation='vertical')
    axs[1, 0].text(-0.2, 0.5, 'Blur', size=25, ha="center", va="center", transform=axs[1, 0].transAxes,
                   rotation='vertical')
    axs[2, 0].text(-0.2, 0.5, 'InvBlur', size=25, ha="center", va="center", transform=axs[2, 0].transAxes,
                   rotation='vertical')

    axs[0, 0].text(0.5, 1.2, 't=0', size=25, ha="center", va="center", transform=axs[0, 0].transAxes,
                   rotation='horizontal')
    axs[0, 1].text(0.5, 1.2, 't=1', size=25, ha="center", va="center", transform=axs[0, 1].transAxes,
                   rotation='horizontal')
    axs[0, 2].text(0.5, 1.2, 't=2', size=25, ha="center", va="center", transform=axs[0, 2].transAxes,
                   rotation='horizontal')
    axs[0, 3].text(0.5, 1.2, 't=3', size=25, ha="center", va="center", transform=axs[0, 3].transAxes,
                   rotation='horizontal')

    plt.show()
