import torch
import numpy as np


def gram(tensor):
    """
        Constructs the Gramian matrix out of the tensor
    """
    return torch.mm(tensor, tensor.t())


def gram_loss(noise_img_gram, style_img_gram, N, M):
    """
        Gramian loss: the SSE between Gramian matrices of a layer
            arXiv:1508.06576v2 - equation (4)
    """
    return torch.sum(torch.pow(noise_img_gram - style_img_gram, 2)).div((np.power(N*M*2, 2, dtype=np.float64)))


def total_variation_loss(image):
    """
        Variation loss makes the images smoother, defined over spacial dimensions
    """
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


def content_loss(noise: torch.Tensor, image: torch.Tensor):
    """
        Simple SSE loss over the generated image and the content image
            arXiv:1508.06576v2 - equation (1)
    """
    return 1/2. * torch.sum(torch.pow(noise - image, 2))
