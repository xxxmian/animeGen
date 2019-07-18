import torch.nn as nn
class NetG(nn.Module):
    def __init__(self, opt):
        super(NetG, self).__init__()
        ngf = opt.ngf
        self.main = nn.Sequential(
            # input is 1x1@100
            nn.ConvTranspose2d(opt.nz, ngf*8, 4, 1, 0),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(),
            # 4x4@512
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),
            # 8x8@256
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),
            # 16x16@128
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            # 32x32@64
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1),
            nn.Tanh() #ToDo 输出范围 -1~1 故而采用Tanh
            # output is 96x96@3
        )
    def forward(self, input):
        return self.main(input)

class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            # 96x96@3
            nn.Conv2d(3, ndf, 5, 3, 1),
            # ToDo 为什么这里不用batchnorm
            nn.ReLU(),
            # 32x32@64
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.ReLU(),
            # 16x16@128
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU(),
            # 8x8@256
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            nn.BatchNorm2d(ndf*8),
            nn.ReLU(),
            # 4x4@512
            nn.Conv2d(ndf*8, 1, 4, 1, 0),
            nn.Sigmoid()
            # output is 1x1@1
        )
    def forward(self, input):
        return self.main(input)