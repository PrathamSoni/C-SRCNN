#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate generated images
"""
from skimage.measure import compare_psnr,compare_ssim 
import scipy
import numpy as np
import glob,os
from itertools import repeat
import pandas as pd

SCALE=3.
folder='Test/CT'

def eval(pred,y):
    ssim=compare_ssim(y,pred)#,data_range=255)
    psnr=compare_psnr(y,pred)
    return psnr,ssim
def generateGT(pred,y,path):
    gt=y[0:pred.shape[0],0:pred.shape[1]]
    scipy.misc.imsave(path+'.GT',gt,format='bmp')
    return gt
def generateBL_bicubic(pred,y,path):#baseline
    bicubic= scipy.ndimage.interpolation.zoom(y, (1./SCALE), prefilter=True)#down-scale
    bicubic= scipy.ndimage.interpolation.zoom(bicubic, (SCALE/1.), prefilter=True)#up-scale
    bicubic= bicubic[0:pred.shape[0],0:pred.shape[1]]
    scipy.misc.imsave(path+'.bicubic',bicubic,format='bmp')
    return bicubic

def generateBL_nn(pred,y,path):#baseline nearest neighbor
    nn= scipy.ndimage.interpolation.zoom(y, (1./SCALE), order=0,mode='nearest',prefilter=True)#down-scale
    nn= scipy.ndimage.interpolation.zoom(nn, (SCALE/1.), order=0,mode='nearest',prefilter=True)#up-scale
    nn= nn[0:pred.shape[0],0:pred.shape[1]]
    scipy.misc.imsave(path+'.nn',nn,format='bmp')
    return nn

def loadImg(filename,mode='org'):
    #print(mode)
    if mode=='c1':
        filename+='.c1'
    elif mode=='c5':
        filename+='.c5'
    elif mode=='c9':
        filename+='.c9'
    #print(filename)
    return scipy.misc.imread(filename, flatten=True, mode='YCbCr').astype(np.uint8)


namelist_org=glob.glob(os.path.join(folder, "*.bmp"))
imglist_org=list(map(loadImg,namelist_org))

imglist_c9=list(map(loadImg,namelist_org,repeat('c9')))
imglist_c5=list(map(loadImg,namelist_org,repeat('c5')))
imglist_c1=list(map(loadImg,namelist_org,repeat('c1')))

imglist_GT=list(map(generateGT,imglist_c9,imglist_org,namelist_org))
imglist_bicubic=list(map(generateBL_bicubic,imglist_c9,imglist_org,namelist_org))
imglist_nn=list(map(generateBL_nn,imglist_c9,imglist_org,namelist_org))

metric_c9=list(map(eval,imglist_c9,imglist_GT))
metric_c5=list(map(eval,imglist_c5,imglist_GT))
metric_c1=list(map(eval,imglist_c1,imglist_GT))
metric_bicubic=list(map(eval,imglist_bicubic,imglist_GT))
metric_nn=list(map(eval,imglist_nn,imglist_GT))
#save
output=np.zeros((len(imglist_GT),10),dtype=np.float32)#each row: psnr BL, psnr c1, psnr c2, psnr c3, ssim BL,ssim c1, ssim c2, ssim c3
output[:,0]=[i[0] for i in metric_nn]
output[:,5]=[i[1] for i in metric_nn]
output[:,1]=[i[0] for i in metric_bicubic]
output[:,6]=[i[1] for i in metric_bicubic]
output[:,2]=[i[0] for i in metric_c1]
output[:,7]=[i[1] for i in metric_c1]
output[:,3]=[i[0] for i in metric_c5]
output[:,8]=[i[1] for i in metric_c5]
output[:,4]=[i[0] for i in metric_c9]
output[:,9]=[i[1] for i in metric_c9]

df=pd.DataFrame(output)
df.columns = ['PSNR_NN','PSNR_BC','PSNR_c1','PSNR_c5','PSNR_c9','SSIM_NN','SSIM_BC','SSIM_c1','SSIM_c5','SSIM_c9']
df['filename']=namelist_org
df.to_csv(os.path.join(folder,'metric.csv'),index=False)
