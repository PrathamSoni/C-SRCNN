#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 04:23:22 2018

@author: ht
"""

import pydicom
import scipy.misc
import glob

def dicom2bmp(filename):
    data=pydicom.dcmread(filename)
    img=data.pixel_array
    img=scipy.misc.imresize(img,(256,256),interp='bicubic')
    outputname=filename.replace('UncompressedDICOMs/','BMP/')
    outputname=outputname.replace('.dcm','.bmp')
    scipy.misc.imsave(outputname,img)
    return True

nameList=glob.glob('CT/UncompressedDICOMs/*.dcm')
result=list(map(dicom2bmp,nameList))