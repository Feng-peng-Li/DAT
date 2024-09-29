import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(torch.nn.Module):

    def __init__(self, in_channels: int = 100, out_channels: int = 3) -> None:
        super(Generator, self).__init__()
        self.fc = torch.nn.Sequential(

            torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=1024, kernel_size=(2, 2), stride=1,
                                     padding=0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(True),
            # output: (1024, 4, 4)

            torch.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=1,
                                     bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            # ouput: (512, 8, 8)

            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            # ouput: (256, 16, 16)

            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            # ouput: (128, 32, 32)

            torch.nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1,
                                     bias=False),
            torch.nn.Tanh()
            # ouput: (3, 64, 64)

            # output shape = (input shape - 1) * stride - 2 * padding + kernel_size
        )

    def forward(self, x):
        output = self.fc(x)
        return output


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Generator_MLP(nn.Module):
    def __init__(self,in_channel,out_channel,img_h,img_w,num_class):
        super(Generator_MLP, self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.h=img_h
        self.w=img_w
        self.feature_emb = nn.Embedding(num_class, num_class)
        self.linear=nn.Linear(num_class,10)
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat,0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.in_channel+num_class, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.out_channel*self.h*self.w),
            nn.Sigmoid()
            )
        self.apply(init_weights)

    def forward(self, z,feat):
        
        z=torch.cat([z,feat],dim=1)
        img = self.model(z)
        img = img.view(img.shape[0], self.out_channel,self.h,self.w)
        return img