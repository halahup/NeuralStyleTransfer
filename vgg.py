import torch.nn as nn
import torchvision.models as models


# define the VGG
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        # load the vgg model's features
        self.vgg = models.vgg19(pretrained=True).features
    
    def get_content_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
            Extracts the features for the content loss from the block4_conv2 of VGG19
            Args:
                x: torch.Tensor - input image we want to extract the features of
            Returns:
                features: torch.Tensor - the activation maps of the block4_conv2 layer
        """
        features = self.vgg[:23](x)
        return features
    
    def get_style_activations(self, x):
        """
            Extracts the features for the style loss from the block1_conv1, 
                block2_conv1, block3_conv1, block4_conv1, block5_conv1 of VGG19
            Args:
                x: torch.Tensor - input image we want to extract the features of
            Returns:
                features: list - the list of activation maps of the block1_conv1, 
                    block2_conv1, block3_conv1, block4_conv1, block5_conv1 layers
        """
        features = [self.vgg[:4](x)] + [self.vgg[:7](x)] + [self.vgg[:12](x)] + [self.vgg[:21](x)] + [self.vgg[:30](x)] 
        return features
    
    def forward(self, x):
        return self.vgg(x)