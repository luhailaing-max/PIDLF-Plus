# -*- coding: utf-8 -*-
"""AfriSAR Canopy Height Script for Pongara
本代码必须首先运行,以为后续PIDLF, PIDLF+准备数据,主要有两个功能：
    * 读取原始SLC数据, 处理结果另存为xxx.h5文件中,后续所有程序均基于该文件执行
    * 基于上一步获取的h5文件, 计算传统方法的森林高度反演结果 
"""
import os
import os.path
import numpy as np
from scipy import ndimage
import kapok.topo
from kapok.lib import calcslope, linesegmentdist
import kapok
# Path and filename of an annotation (.ann) file in the downloaded UAVSAR
nptodb = 20/np.log(10)
def main(ti, site, method):

    print("begain --------------%s--------------"%ti)
    if site == 'rabifo':
        inputann = r'E:\datasets\dataset4-Rabifo_TM130\rabifo_TM130_16010_000_160228_L090HH_01_BC.ann' 
        # Name of the Kapok HDF5 file to create containing the covariance matrix,
        kapokfile = 'rabifo_kapok.h5' # lope_kapok.h5, pongara_kapok.h5, rabifo_kapok.h5
    elif site == 'lope':
        inputann = r'E:\datasets\AfriSAR2016\lopenp_TM140_16008_002_160225_L090HH_03_BC.ann' 
        # Name of the Kapok HDF5 file to create containing the covariance matrix,
        kapokfile = 'lope_kapok.h5' # lope_kapok.h5, pongara_kapok.h5, rabifo_kapok.h5
    elif site == "pongara":
        inputann = r'E:\datasets\dataset3-Pongara_TM275\pongar_TM275_16009_002_160227_L090HH_01_BC.ann' 
        # Name of the Kapok HDF5 file to create containing the covariance matrix,
        kapokfile = 'pongara_kapok.h5' # lope_kapok.h5, pongara_kapok.h5, rabifo_kapok.h5

    savepath = r'E:\codes\SAR2' # save for .h5 file
    outputpath =  r'E:\codes\SAR2\output\Traditional' # height map out path
    outputpath = os.path.join(outputpath, method)

    # Size of the multilooking window to use when estimating the covariance
    # matrix from the SLC data, in (azimuth, range) dimensions.
    mlwin = (20,5)
    # If you wish to use the .kz files downloaded as part of the UAVSAR SLC stack,
    # set calculate_flat_kz = False.  If you wish to have Kapok calculate
    # the kz assuming a flat earth, without terrain correction, set
    # calculate_flat_kz = True.  For the Pongara dataset, we set
    # calculate_flat_kz to True, as that area has relatively minor topographic
    # variation.  For the Lope dataset, we set calculate_kz = False, as that area
    # displays significant topography.  This option also determines if the
    # terrain slope is used in the forest height estimation.
    calculate_flat_kz = False

    # A fixed extinction value to use for inversion of the random volume over
    # ground model, if desired.  If you wish to solve for the extinction as a
    # free parameter, set ext = None (default).  Note that Kapok uses Np/m as the
    # units for the extinction parameter, so if you wish to use dB/m, you must
    # divide by nptodb, which is equal to 20/log(10).  For example:
    # ext = 0.1/nptodb # 0.1 dB/m fixed extinction
    ext = None

    #############
    # LOAD DATA #
    #############

    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
        # os.makedirs(savepath+'supplementary_products/')

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    # Load the Kapok HDF5 file if it already exists.  If not, import the UAVSAR
    # SLC stack to create it.
    # kapokfile = savepath + kapokfile
    kapokfile = os.path.join(savepath, kapokfile)
    # if os.path.isfile(kapokfile):
    if os.path.exists(kapokfile):
        import kapok
        scene = kapok.Scene(kapokfile)
    else:
        import kapok.uavsar
        # Note: num_blocks in the below function call can be increased if you
        # have memory problems loading in the data.
        scene = kapok.uavsar.load(inputann,kapokfile,mlwin=mlwin,num_blocks=50,kzcalc=calculate_flat_kz)

    scene.opt()

    # Now, we create a mask which identifies low HV backscatter areas.
    mask = scene.power('HV') # Get the HV backscattered power (in linear units).
    mask[mask <= 0] = 1e-10 # Get rid of zero-valued power.
    mask = 10*np.log10(mask) # Convert to dB.
    mask = mask > -22 # Find pixels above/below -22 dB threshold.

    # If this mask is provided to the model inversion, only pixels with HV
    # sigma-nought over -22 dB will be considered valid pixels for the forest
    # height estimation.  This will also save some computation time, since
    # these pixels will be skipped over by the algorithm.

    #################################
    # CREATE CANOPY HEIGHT PRODUCTS #
    #################################

    epsilon = 0.4

    # 以下每个反演方法依次执行

    print(".......%s model inversion...."%method)

    canopyheight_dataset_name = method

    # Should we correct the model inversion for the range terrain slope angle?
    if calculate_flat_kz:
        rangeslope = None
    else:
        rangeslope, azimuthslope = calcslope(scene.dem, scene.spacing, scene.inc)

    # Perform the random volume over ground model inversion to estimate the canopy
    # heights and other parameters.
    # Note: If the RVoG inversion has already been run, the next line of code will
    # not overwrite it!  If you wish to overwrite a previous run, set the
    # overwrite keyword in the line below to True.

    # 
    if method == 'rvog':
        canopyheightdataset = scene.inv(method='rvog', name=canopyheight_dataset_name, desc='Sigle-baseline rvog model inversion.',
                            bl=0, blcriteria='line', hv_min=0, hv_max=60, ext=ext, mask=mask, rngslope=rangeslope, overwrite=True)
        canopyheight = scene.get(canopyheight_dataset_name+'/hv')
    elif method == 'sinc':
        canopyheightdataset = scene.inv(method='sinc', name=canopyheight_dataset_name, desc='Sigle-baseline sinc model inversion.',
                            bl=0, blcriteria='line', mask=mask,  overwrite=True)
        canopyheight = scene.get(canopyheight_dataset_name+'/hv')
    elif method == 'sincphase':
        canopyheightdataset = scene.inv(method='sincphase', name=canopyheight_dataset_name, desc='Sigle-baseline sincphase model inversion.',
                            bl=0, blcriteria='line', mask=mask, rngslope=rangeslope, overwrite=True)
        canopyheight = scene.get(canopyheight_dataset_name+'/hv')
    elif method =='dem':
        canopyheightdataset = scene.inv(method='sincphase', name=canopyheight_dataset_name, desc='Sigle-baseline sincphase model inversion.',
                            bl=0, blcriteria='line',   mask=mask, rngslope=rangeslope, overwrite=True)

        canopyheightdataset = scene.inv(method='sincphase', name=canopyheight_dataset_name, desc='Sigle-baseline sincphase model inversion.',
                            bl=0, blcriteria='line',  mask=mask, rngslope=rangeslope, overwrite=True)
        sincphasehv = scene.get('sincphase'+'/hv')
        sinchv = scene.get('sinc'+'/hv')
        canopyheight = sincphasehv - epsilon*sinchv

    # if method == 'rvog':
    #     canopyheight = scene.get(canopyheight_dataset_name+'/hv')
    # elif method == 'sinc':
    #     canopyheight = scene.get(canopyheight_dataset_name+'/hv')
    # elif method == 'sincphase':
    #     canopyheight = scene.get(canopyheight_dataset_name+'/hv')
    # elif method =='dem':
    #     sincphasehv = scene.get('sincphase'+'/hv')
    #     sinchv = scene.get('sinc'+'/hv')
    #     canopyheight = sincphasehv - epsilon*sinchv


    # Get array of canopy heights.
    # canopyheight = scene.get(canopyheight_dataset_name+'/hv')

    # Export geocoded canopy heights.
    # canopyheight[canopyheight < 2] = 0

    outdir = os.path.join(outputpath,str(ti)+site+'_'+method+'.tif') 
    scene.geo(canopyheight,outdir,nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

if __name__ == "__main__":
    N = 1 # run N times per method, default: 1
    method = 'dem' # sinc,  dem, rvog
    site = 'rabifo' # lope, pongara, rabifo
    for ti in range(N):
        main(ti, site, method)