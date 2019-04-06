import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch import optim
from torchvision.utils import save_image
from vgg import VGG
from losses import gram, gram_loss, total_variation_loss, content_loss
import os


parser = argparse.ArgumentParser()
parser.add_argument("--style_img", help="Path of the style image", type=str)
parser.add_argument("--content_img", help="Path of the content image", type=str)
parser.add_argument("--img_dim", help="Dimensions of the resulting image", type=int, default=512)
parser.add_argument("--num_iter", help="Number of iterations to run", type=int, default=10000)
parser.add_argument("--style_loss_weight", help="Weight for the style loss", type=int, default=10e6)
parser.add_argument("--content_loss_weight", help="Weight for the content loss", type=int, default=10e-4)
parser.add_argument("--variation_loss_weight", help="Weight for the variation loss", type=int, default=10e3)
parser.add_argument("--print_every", help="Print the progress every K iretations", type=int, default=1000)
parser.add_argument("--save_every", help="Save the image every K iterations", type=int, default=100)


def main(style_img_path: str,
         content_img_path: str, 
         img_dim: int,
         num_iter: int,
         style_weight: int,
         content_weight: int,
         variation_weight: int,
         print_every: int,
         save_every: int):

    assert style_img_path is not None
    assert content_img_path is not None

    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # read the images
    style_img = Image.open(style_img_path)
    cont_img = Image.open(content_img_path)
    
    # define the transform
    transform = transforms.Compose([transforms.Resize((img_dim, img_dim)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    # get the tensor of the image
    content_image = transform(cont_img).unsqueeze(0).to(device)
    style_image = transform(style_img).unsqueeze(0).to(device)
    
    # init the network
    vgg = VGG().to(device).eval()
    
    # replace the MaxPool with the AvgPool layers
    for name, child in vgg.vgg.named_children():
        if isinstance(child, nn.MaxPool2d):
            vgg.vgg[int(name)] = nn.AvgPool2d(kernel_size=2, stride=2)
            
    # lock the gradients
    for param in vgg.parameters():
        param.requires_grad = False
    
    # get the content activations of the content image and detach them from the graph
    content_activations = vgg.get_content_activations(content_image).detach()
    
    # unroll the content activations
    content_activations = content_activations.view(512, -1)
    
    # get the style activations of the style image
    style_activations = vgg.get_style_activations(style_image)
    
    # for every layer in the style activations
    for i in range(len(style_activations)):

        # unroll the activations and detach them from the graph
        style_activations[i] = style_activations[i].squeeze().view(style_activations[i].shape[1], -1).detach()

    # calculate the gram matrices of the style image
    style_grams = [gram(style_activations[i]) for i in range(len(style_activations))]
    
    # generate the Gaussian noise
    noise = torch.randn(1, 3, img_dim, img_dim, device=device, requires_grad=True)
    
    # define the adam optimizer
    # pass the feature map pixels to the optimizer as parameters
    adam = optim.Adam(params=[noise], lr=0.01, betas=(0.9, 0.999))

    # run the iteration
    for iteration in range(num_iter):

        # zero the gradient
        adam.zero_grad()

        # get the content activations of the Gaussian noise
        noise_content_activations = vgg.get_content_activations(noise)

        # unroll the feature maps of the noise
        noise_content_activations = noise_content_activations.view(512, -1)

        # calculate the content loss
        content_loss_ = content_loss(noise_content_activations, content_activations)

        # get the style activations of the noise image
        noise_style_activations = vgg.get_style_activations(noise)

        # for every layer
        for i in range(len(noise_style_activations)):

            # unroll the the noise style activations
            noise_style_activations[i] = noise_style_activations[i].squeeze().view(noise_style_activations[i].shape[1], -1)

        # calculate the noise gram matrices
        noise_grams = [gram(noise_style_activations[i]) for i in range(len(noise_style_activations))]

        # calculate the total weighted style loss
        style_loss = 0
        for i in range(len(style_activations)):
            N, M = noise_style_activations[i].shape[0], noise_style_activations[i].shape[1]
            style_loss += (gram_loss(noise_grams[i], style_grams[i], N, M) / 5.)

        # put the style loss on device
        style_loss = style_loss.to(device)
            
        # calculate the total variation loss
        variation_loss = total_variation_loss(noise).to(device)

        # weight the final losses and add them together
        total_loss = content_weight * content_loss_ + style_weight * style_loss + variation_weight * variation_loss

        if iteration % print_every == 0:
            print("Iteration: {}, Content Loss: {:.3f}, Style Loss: {:.3f}, Var Loss: {:.3f}".format(iteration, 
                                                                                                     content_weight * content_loss_.item(),
                                                                                                     style_weight * style_loss.item(), 
                                                                                                     variation_weight * variation_loss.item()))

        # create the folder for the generated images
        if not os.path.exists('./generated/'):
            os.mkdir('./generated/')
        
        # generate the image
        if iteration % save_every == 0:
            save_image(noise.cpu().detach(), filename='./generated/iter_{}.png'.format(iteration))

        # backprop
        total_loss.backward()
        
        # update parameters
        adam.step()
        

if __name__ == "__main__":
    
    # parse arguments
    args = parser.parse_args()
    style_img_path = args.style_img
    content_img_path = args.content_img
    img_dim = args.img_dim
    num_iter = args.num_iter
    style_weight = args.style_loss_weight
    content_weight = args.content_loss_weight
    variation_weight = args.variation_loss_weight
    print_every = args.print_every
    save_every = args.save_every
    
    # run the model
    main(style_img_path=style_img_path,
         content_img_path=content_img_path,
         img_dim=img_dim,
         num_iter=num_iter,
         style_weight=style_weight,
         content_weight=content_weight,
         variation_weight=variation_weight,
         print_every=print_every,
         save_every=save_every)
