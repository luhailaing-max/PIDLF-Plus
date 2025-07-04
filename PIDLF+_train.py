# -*- coding: utf-8 -*-
# v1-PIDLF + loss

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import os.path
import kapok
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler

from torch.nn import functional as F
from pcgrad import PCGrad
import time

import random
import torch.optim as optim
from torch import  nn, einsum
from sklearn.linear_model import LinearRegression
import ResUnetPlusPlus
import gc


def getimgblock(arr, idx, partrow, partcol):
    band, r, c = arr.shape
    rnum = r / partrow
    cnum = c / partcol
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
    def __init__(self, hh, hv, vv, dem, coh,kz, patchrow, patchcol):
  
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
        # showimg(x[0])

        dem = torch.tensor(dem).unsqueeze(0)
        coh = torch.tensor(coh).unsqueeze(0)
        b, self.h, self.w = coh.shape

        self.x = padding(x,patchrow,patchcol)
        self.dem = padding(dem,patchrow,patchcol)
        self.coh = padding(coh,patchrow,patchcol)

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
            'w':self.w
                  }
        return output

def showimg(arr):
    plt.figure(' arr')
    plt.imshow(arr)
    plt.colorbar(shrink = 0.8)
    plt.show()
    print("finished....")

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

def main(site, idx):
    import kapok
    print("begain --------------%s--------------"%idx)

    # For the Pongara dataset has relatively minor topographic, , terrain = 0.
    # For the Lope dataset have significant topography, terrain = 1.


    if site == 'rabifo': # lope, pongara, rabifo
        datafile = '/home/hllu/codes/SAR2/rabifo_kapok.h5'
    elif site == 'lope': # lope, pongara, rabifo
        datafile = '/home/hllu/codes/SAR2/lope_kapok2.h5'
    elif site == 'pongara':
        datafile = '/home/hllu/codes/SAR2/pongara_kapok.h5'

    logpath =os.path.join('/home/hllu/codes/SAR2/output/PIDLF+/', site)

    # First index is the azimuth size, second index is the range size.

    # mlwin = [20,5]

    throd = 60


    # if site !='lope':
    #     scenelope = kapok.Scene(datafilelope)
    

    scene = kapok.Scene(datafile)


    def getdata(scene,bl):
        hh = scene.coh('HH', bl=bl)
        hv = scene.coh('HV', bl=bl)
        vv = scene.coh('VV', bl=bl)
        coh = scene.get('sinc/hv')
        dem = scene.get('dem/hv')
        return hh, hv, vv, coh, dem


    hh, hv, vv, coh, dem = getdata(scene, bl=0)
    # return
     # get kz
    kz = scene.kz(0)

    coh = np.array(coh)
    coh[coh>throd] = throd
    dem = np.array(dem)
    dem[dem>throd] = throd


    patchsize = 64
    logfrequence = 10
    batch_size = 64
    inchannels = 7
    epochs = 120
    # 42, 3407, 114514
    seed = 42
    savemodel_frequence = 120
    sgdlr = 0.0001
    # sgdlr = 0.00001

    momentem = 0.9

   
    dataset = MyDataset(hh,hv,vv,coh,dem,kz, patchrow= patchsize, patchcol= patchsize)

    
    # times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    # logpath = os.path.join(logpath,str(times))
    print("   logpath is : %s"%logpath)
    # checkpoint_path = os.path.join(logpath, checkpoints_name)


    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    a = -8.1433
    b = 1.6172
    lamb = a * np.std(kz) + b
   
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    model = ResUnetPlusPlus.build_resunetplusplus(inc=inchannels).to(device)

    # model = nn.DataParallel(model)
    # model.to('cuda')

    # para=[device, epochs, logfrequence, patchsize, batch_size, datafile,sgdlr, site, lamb, seed]
    # print(para)


    # print(model)
    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        # model = torch.nn.DataParallel(model, device_ids=[1])
        model = torch.nn.DataParallel(model)

    criterion = Loss() #单独一个标签

    #optimizer = optim.SGD(model.parameters(), lr=sgdlr,momentum=momentem,weight_decay=0.000001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    from mySGD import SGD_GC, SGD_GCC
    # optimizer = SGD_GC(model.parameters(), lr=sgdlr, momentum=momentem, weight_decay=0.000001)
    optimizer = SGD_GCC(model.parameters(), lr=sgdlr, momentum=momentem, weight_decay=0.000001)

    # optimizer = PCGrad(optimizer)
    optimizer = PCGrad(optimizer)

        # print("train loader len:...",len(train_loader))
        # writer= SummaryWriter(logpath)
    model.train()
    for epoch in range(epochs):

        # if (epoch+1)%30000 == 0:
        #     sgdlr = sgdlr/10
        #     optimizer = optim.SGD(model.parameters(), lr=sgdlr, momentum=momentem, weight_decay=0.000001)
        totalloss = 0.

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            x = data["x"].to(device).to(torch.float32)
            y1 = data["coh"].to(device).to(torch.float32) # coh
            y2 = data["dem"].to(device).to(torch.float32) # DEM
            # y1 = y1.unsqueeze(1)
            # y2 = y2.unsqueeze(1)
            out= model(x)
        
            loss1 = criterion(out,y1)
            loss2 = criterion(out,y2)

            optimizer.pc_backward([lamb*loss1, loss2])
            # sum(losses).backward()
            optimizer.step()
        
            totalloss = totalloss + loss1 + loss2
        if (epoch + 1) % logfrequence== 0:
            times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            print("    %s:epoch %s/%s-avgtra-loss:%s" % (times, (epoch+1), epochs, str(totalloss / (len(train_loader)))))
            # writer.add_scalar('avg-trainloss', totalloss / (len(train_loader)), global_step=epoch)

        if (epoch + 1) % savemodel_frequence == 0:

            if not os.path.exists(logpath):
                os.makedirs(logpath)
            pathname = str(idx)+str(epoch + 1) + "epoch.pth"

            # on local computer
            torch.save(model.state_dict(), os.path.join(logpath, pathname))
            
            # on kaggle
            # torch.save(model.state_dict(), ("./"+pathname))

            times = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            print("    %s:save %s successfully" % (times, pathname))
    
    # fished_time = time.process_time()
    # program_run_time= fished_time - start_time
  
    # print("finished time.....: %s" % (fished_time))
    # print("    run time.....: %s" % (program_run_time))
    del model
    del dataset
    del train_loader
    del scene
    gc.collect()
    torch.cuda.empty_cache()
    print("finished...............")

if __name__ == "__main__":
    site = 'lope' # lope, pongara, rabifo
    N = 1
    for idx in range(0,N):
        main(site, idx)
















