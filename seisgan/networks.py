import torch
import torch.nn as nn


def get_activation():
    return nn.ReLU()


class LatentInputLayer(nn.Module):
    def __init__(self):
        super(LatentInputLayer, self).__init__()

    def forward(self, z):
        return z*1.


class MaulesCreekVelocity(nn.Module):
    def __init__(self, generator):
        super(MaulesCreekVelocity, self).__init__()
        self.latent_input = LatentInputLayer()
        self.generator = generator
        self.pad_top = nn.ConstantPad2d((0, 0, 32, 0), 2.)
        self.pad_bottom = nn.ConstantPad2d((0, 0, 0, 32), 2.)

    def forward(self, z):
        z = self.latent_input(z)
        x = self.generator(z)
        x = ((x[0, 0, :, :, 8]/2.+0.5)+1)*2
        x = self.pad_top(x)
        x = self.pad_bottom(x)
        x = x.transpose(1, 0).unsqueeze(0).unsqueeze(0).pow(-2)
        return x


class HalfChannels(nn.Module):
    def __init__(self, generator, min_vp, max_vp, vp_top=2., vp_bottom=2., top_size=32, bottom_size=32):
        super(HalfChannels, self).__init__()
        self.min_vp, self.max_vp = min_vp, max_vp

        self.latent_input = LatentInputLayer()
        self.generator = generator
        self.pad_top = nn.ConstantPad2d((0, 0, top_size, 0), vp_top)
        self.pad_bottom = nn.ConstantPad2d((0, 0, 0, bottom_size), vp_bottom)

    def forward(self, z):
        z = self.latent_input(z)
        x_geo = self.generator(z)
        x = (x_geo[:, 1, :, :]/2.+0.5)*(self.max_vp-self.min_vp)+self.min_vp
        x = self.pad_top(x)
        x = self.pad_bottom(x)
        x = x.transpose(2, 1).unsqueeze(0).pow(-2) #convert to square slowness
        return x, x_geo

class HalfChannelsTest(nn.Module):
    def __init__(self, min_vp, max_vp, vp_top=2., vp_bottom=2., top_size=32, bottom_size=32):
        super(HalfChannelsTest, self).__init__()
        self.min_vp, self.max_vp = min_vp, max_vp

        self.pad_top = nn.ConstantPad2d((0, 0, top_size, 0), vp_top)
        self.pad_bottom = nn.ConstantPad2d((0, 0, 0, bottom_size), vp_bottom)

    def forward(self, x_geo):
        x = x_geo[:, 1, :, :]*(self.max_vp-self.min_vp)+self.min_vp
        x = self.pad_top(x)
        x = self.pad_bottom(x)
        x = x.transpose(2, 1).unsqueeze(0).pow(-2) #convert to square slowness
        return x, x_geo

class GeneratorMultiChannel(nn.Module):
    def __init__(self):
        super(GeneratorMultiChannel, self).__init__()
        self.network = self.build_network()
        self.activation_facies = nn.Tanh()
        self.activation_rho = nn.Softplus()

    def build_network(self, activation=get_activation):
        blocks = []
        blocks += [nn.Conv2d(50, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), activation()]
        blocks += [nn.PixelShuffle(upscale_factor=2)]
        blocks += [nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*blocks)

    def forward(self, z):
        x = self.network(z)
        a = self.activation_facies(x[:, 0]).unsqueeze(1)
        b = self.activation_facies(x[:, 1]).unsqueeze(1)
        c = self.activation_rho(x[:, 2]).unsqueeze(1)
        return torch.cat([a, b, c], 1)

class DiscriminatorUpsampling(nn.Module):
    def __init__(self):
        super(DiscriminatorUpsampling, self).__init__()
        self.network = self.build_network()

    def build_network(self, activation=get_activation):
        blocks = []
        blocks += [nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2), activation()]
        blocks += [nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        dec = self.network(x)
        dec = dec.view(-1, 2 * 2 * 2)
        return dec

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.network = self.build_network()

    def build_network(self, activation=get_activation):
        blocks = []
        blocks += [nn.ConvTranspose3d(20, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), activation()]
        blocks += [nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), activation()]
        blocks += [nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), activation()]
        blocks += [nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), activation()]
        blocks += [nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1), nn.Tanh()]
        return nn.Sequential(*blocks)

    def forward(self, z):
        return self.network(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = self.build_network()

    @staticmethod
    def build_network(activation=get_activation):
        blocks = []
        blocks += [nn.Conv3d(3, 32, kernel_size=5, stride=2, padding=1), activation()]
        blocks += [nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv3d(128, 1, kernel_size=3, stride=2, padding=1), activation()]
        return nn.Sequential(*blocks)

    def forward(self, x):
        dec = self.network(x)
        dec = dec.view(-1, 2 ** 4)
        return dec

class DiscriminatorUpsampling(nn.Module):
    def __init__(self):
        super(DiscriminatorUpsampling, self).__init__()
        self.network = self.build_network()

    def build_network(self, activation=get_activation):
        blocks = []
        blocks += [nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2), activation()]
        blocks += [nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), activation()]
        blocks += [nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        dec = self.network(x)
        dec = dec.view(-1, 2 * 2 * 2)
        return dec