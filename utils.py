import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
class Config(object):
    batch_size = 64
    data_path = '/Users/zhangqi/Documents/深度网络技术与应用/DLHW/二次元头像生成/testface'
    workers = 4
    epoches = 2
    img_size = 96
    lr_G = 2e-4
    lr_D = 2e-4
    nz = 100
    ngf = 64
    ndf = 64
    g_every = 5
    d_every = 1
    save_every = 10
    netD_path = None
    netG_path = None
    gpu = False

class FaceDataset(data.Dataset):
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, item):
        img = self.imgs[item]
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.imgs)
def faceDataLoader(opt):
    imgs = []
    for filename in os.listdir(opt.data_path):
        imgs.append(Image.open(opt.data_path+'/'+filename))
    dataset = FaceDataset(imgs, transforms.Compose([
        transforms.Resize(opt.img_size),
        transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))
    dataloader = data.DataLoader(dataset,batch_size=opt.batch_size, num_workers=opt.workers)
    return dataloader


