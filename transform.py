import math
import torch
import torchvision.transforms as transform
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode as Interpolation



def get_transform():
   
    transform_train1 = transform.Compose([
            transform.RandomCrop(32, padding=4),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
        ])
    
        
   
    transform_test = transform.Compose([
            transform.ToTensor(),
        ])

    transform_train = [transform_train1, transform_test]
    return transform_train,transform_test

class transfrom_set(torch.nn.Module):
    def __init__(self):
        super().__init__()

        layers = [
                transform.RandomHorizontalFlip(),
                RandomCrop(0, 11),
                transform.ToTensor(),
                transform.RandomErasing(p=0.5)
            ]
        self.layers = transform.Compose(layers)

    def forward(self, img):
        return self.layers(img)
    

class RandomCrop(torch.nn.Module):
    def __init__(self, low, high=None):
        super().__init__()
        high = low if high is None else high
        self.low, self.high = int(low), int(high)

    def sample_top(self, x, y):
        x = torch.randint(0, x + 1, (1,)).item()
        y = torch.randint(0, y + 1, (1,)).item()
        return x, y

    def forward(self, img):
        if self.low == self.high:
            strength = self.low
        else:
            strength = torch.randint(self.low, self.high, (1,)).item()

        w, h = F.get_image_size(img)
        crop_x = torch.randint(0, strength + 1, (1,)).item()
        crop_y = strength - crop_x
        crop_w, crop_h = w - crop_x, h - crop_y

        top_x, top_y = self.sample_top(crop_x, crop_y)

        img = F.crop(img, top_y, top_x, crop_h, crop_w)
        img = F.pad(img, padding=[crop_x, crop_y], fill=0)

        top_x, top_y = self.sample_top(crop_x, crop_y)

        return F.crop(img, top_y, top_x, h, w)