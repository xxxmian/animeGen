from utils import *
import torch.nn as nn
import torch
import model
import tqdm
from torchnet.meter import AverageValueMeter
from torchvision import transforms
import matplotlib.pyplot as plt
def train():
    opt = Config()
    device = torch.device('cuda') if opt.gpu else torch.device('cpu')

    dataloader = faceDataLoader(opt)

    netG = model.NetG(opt).to(device)
    netD = model.NetD(opt).to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(netG.parameters(), opt.lr_G)
    optimizer_D = torch.optim.Adam(netD.parameters(), opt.lr_D)

    errD = AverageValueMeter()
    errG = AverageValueMeter()

    real_label = torch.ones(opt.batch_size).to(device)
    fake_label = torch.zeros(opt.batch_size).to(device)
    fix_noise = torch.randn(1, opt.nz, 1, 1)

    for epoch in range(opt.epoches):
        for i, input in tqdm.tqdm(enumerate(dataloader)):
            # optimize Discriminator first
            if input.shape[0] != opt.batch_size:
                continue
            if (i+1) % opt.d_every == 0:
                netG.eval()
                netD.train()
                optimizer_D.zero_grad()
                # real images scores 1
                output = netD(input)
                err_real = criterion(output, real_label)
                err_real.backward()
                # fake images scores 0
                noise = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
                output = netD(netG(noise))
                err_fake = criterion(output, fake_label)
                err_fake.backward()

                errD.add((err_real+err_fake).item())
                optimizer_D.step()
            # optimize Generator
            if (i+1) % opt.g_every == 0:
                netD.eval()
                netG.train()
                optimizer_G.zero_grad()
                noise = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
                err_g = criterion(netD(netG(noise)),real_label)
                err_g.backward()
                errG.add(err_g.item())
                optimizer_G.step()
                netD.train()
        if (epoch+1) % 1 == 0:
            imgs = netG(fix_noise)
            imgs = imgs.cpu().squeeze(0)
            imgs = transforms.ToPILImage()(imgs[:])
            imgs.convert('RGB')
            plt.imshow(imgs)
            plt.show()


        if (epoch+1) % opt.save_every == 0:
            torch.save(netD.state_dict(), 'checkpoints/netD_%s.pth' % epoch)
            torch.save(netG.state_dict(), 'checkpoints/netG_%s.pth' % epoch)

if __name__ == '__main__':
    train()