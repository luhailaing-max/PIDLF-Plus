# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import os.path
import kapok
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from torch.nn import functional as F
from pcgrad import PCGrad
import time
import random
import torch.optim as optim
from torch import  nn, einsum
import ResUnetPlusPlus
import math

def getimgblock(arr, idx, partrow, partcol):
    band, r, c = arr.shape
    # rnum = r / partrow
    # cnum = c / partcol
    rnum = math.ceil(r / partrow)
    cnum = math.ceil(c / partcol)
    tem = idx
    idr = int(tem // cnum)
    idc = int(tem % cnum)
    idrstart = partrow * idr
    idrend = partrow * idr + partrow
    idcstart = partcol * idc
    idcend = partcol * idc + partcol

    img= arr[:, idrstart:idrend, idcstart:idcend]
    return img


def padding(arr, partrow, partcol):
    band, r, c = arr.shape
    # print("padding before %s"%str(arr.shape))
    if r % partrow == 0:
        row = r
    else:
        row = r + (partrow - r % partrow)
    if c % partcol == 0:
        col = c
    else:
        col = c + (partcol - c % partcol)
    rowp = row - r
    colp = col - c
    arr = np.pad(arr, ((0, 0), (0, rowp), (0, colp)), "constant")
    # print("padding after %s"%str(arr.shape))
    return arr

class MyDataset(Dataset):
    def __init__(self, hh, hv, vv, dem, coh, kz,patchrow, patchcol):
  
        hh1 = hh.real
        hh2 = hh.imag
        hv1 = hv.real
        hv2 = hv.imag
        vv1 = vv.real
        vv2 = vv.imag

        hh1 = torch.tensor(hh1).unsqueeze(0)
        hv1 = torch.tensor(hv1).unsqueeze(0)
        vv1 = torch.tensor(vv1).unsqueeze(0)

        hh2 = torch.tensor(hh2).unsqueeze(0)
        hv2 = torch.tensor(hv2).unsqueeze(0)
        vv2 = torch.tensor(vv2).unsqueeze(0)

        kz = torch.tensor(kz).unsqueeze(0)
        x = torch.cat((hh1,hh2,hv1,hv2,vv1,vv2,kz), 0)

        dem = torch.tensor(dem).unsqueeze(0)
        coh = torch.tensor(coh).unsqueeze(0)
        b, self.h, self.w = coh.shape

        self.x = padding(x,patchrow,patchcol)
        self.dem = padding(dem,patchrow,patchcol)
        self.coh = padding(coh,patchrow,patchcol)

        _, self.pah, self.paw = self.coh.shape
        self.patchrow = patchrow
        self.patchcol = patchcol
      
        # showimg(self.x[0])

    def __len__(self):
        band, r, c = self.x.shape
        rnum = r / self.patchrow
        cnum = c / self.patchcol
        num = int(rnum * cnum)

        return num

    def __getitem__(self, idx):
        # print("idx:%s"%idx)
        # hh = getimgblock(self.hh,idx,self.patchrow,self.patchcol)
        # hv = getimgblock(self.hv,idx,self.patchrow,self.patchcol)
        # vv = getimgblock(self.vv,idx,self.patchrow,self.patchcol)
        x = getimgblock(self.x,idx,self.patchrow,self.patchcol)
        dem = getimgblock(self.dem,idx,self.patchrow,self.patchcol)
        coh = getimgblock(self.coh,idx,self.patchrow,self.patchcol)
    
        output = {
            "idx":idx,
            "x": x,
            "dem": dem,
            "coh": coh,
            'patchsize': self.patchcol,
            'h': self.h,
            'w':self.w,
            'pah':self.pah,
            'paw': self.paw,
    
                  }
        return output

def showimg(arr):
    plt.figure(' arr')
    plt.imshow(arr)
    plt.colorbar(shrink = 0.8)
    plt.show()
    print("finished....")


class SingleModel(nn.Module):
    def __init__(self,inchannel, outchannel):
        super(SingleModel, self).__init__()
        self.con1 = nn.Conv2d(inchannel, 32, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.con2 = nn.Conv2d(32, 16, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.con3 = nn.Conv2d(16,outchannel, kernel_size=5, padding=2)
        self.out = nn.Identity()

    def forward(self,x):
        x = self.con1(x)
        x = self.relu1(x)
        x = self.con2(x)
        x = self.relu2(x)
        x = self.con3(x)
        x = self.out(x)
        return x

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    def forward(self,x,y):
        loss = F.mse_loss(x,y)
        return loss

def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    np.random.seed(seed) # Numpy module.
    random.seed(seed) # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(ti, site):
    print("begain --------------%s--------------"%ti)
    import kapok
    if site == 'rabifo': # lope, pongara, rabifo
        datafile = r'E:\codes\SAR2\rabifo_kapok.h5'
    elif site == 'lope': # lope, pongara, rabifo
        datafile = r'E:\codes\SAR2\lope_kapok.h5'
    elif site == 'pongara':
        datafile = r'E:\codes\SAR2\pongara_kapok.h5'

    outpath = os.path.join(r'E:\codes\SAR2\output\PIDLF+', site)
    logpath =os.path.join(r'E:\codes\SAR2\output\PIDLF+', site)

    checkpoints_name =str(ti)+"120epoch.pth"

    method = "PIDLF+"
    checkpoint_path = os.path.join(logpath, checkpoints_name)

    patchsize = 64
    throd = 60
    seed = 42

    # First index is the azimuth size, second index is the range size.
    mlwin = [20,5]

    if not os.path.exists(outpath):
        os.makedirs(outpath)


    scene = kapok.Scene(datafile)


    def getdata(scene,bl):
        hh = scene.coh('HH', bl=bl)
        hv = scene.coh('HV', bl=bl)
        vv = scene.coh('VV', bl=bl)
        coh = scene.get('sinc/hv')
        dem = scene.get('dem/hv')
        return hh, hv, vv, coh, dem

    kz = scene.kz(0)

    hh, hv, vv, coh, dem = getdata(scene, bl=0)
    print("getdata finished.....")
    
    coh = np.array(coh)
    coh[coh>throd] = throd

    dem = np.array(dem)

    dem[dem>throd] = throd


    dataset = MyDataset(hh,hv,vv,coh,dem,kz, patchrow= patchsize, patchcol= patchsize)

    batch_size = 4

    inchannels = 7
    # print("in channels is :",channels)


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed_everything(seed)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)


    model = ResUnetPlusPlus.build_resunetplusplus(inc=inchannels).to(device)

    if os.path.exists(checkpoint_path):
        # model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        # model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint_path, map_location=torch.device(device)).items()})
        print("    Success to loading model dict from %s ....."%checkpoint_path)
    else:
        print("    Failed to load model dict  from %s ....."%checkpoint_path)
        return



    temdata = dataset.__getitem__(0)
    realr = temdata['h']
    realc = temdata['w']
    row = temdata['pah']
    col = temdata['paw']
    img = torch.zeros((1,row,col))
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(train_loader):
            idxx = data["idx"].to(device)

            x = data["x"].to(device).to(torch.float32)
            out= model(x)
            f2 = torch.squeeze(out)
            for i, f2i in enumerate(f2):
                rnum = row / patchsize
                cnum = col / patchsize
                #banchsize>1时使用以下语句
                tem = idxx[i]
                # tem = idxx
                idr = int(tem // cnum)
                idc = int(tem % cnum)
                idrstart = patchsize * idr
                idrend = patchsize * idr + patchsize
                idcstart = patchsize * idc
                idcend = patchsize * idc + patchsize
        
                #  无重叠或者重叠区域直接覆盖
                img[:, idrstart:idrend, idcstart:idcend] = f2i

    out = img[:, 0:realr, 0:realc]
    out = torch.squeeze(out).numpy()
    # out = out.astype('int16')
    out[out>throd] =throd
    out[out<0] = 0

    # outmask = out + 1
   
    # outmask[outmask>1] = 0
    # out = out + dem * outmask
    outdir = os.path.join(outpath,str(ti)+site+'_'+method+'.tif') 
    scene.geo(out,outdir,nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

    print("finished...............")



if __name__ == "__main__":
    N = 10
    site = 'lope' # lope, pongara, rabifo
    for ti in range(1,N):
        main(ti, site)

































