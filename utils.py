"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import scipy.misc
import scipy.ndimage
import numpy as np
from functools import reduce

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
"""7-1-2 load h5"""
def read_data(path):
    """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
    """  
    #print('check1')
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'),dtype=np.float16)
        try:
            label = np.array(hf.get('label'),dtype=np.float16)
            return data, label
        except:
            return data

"""7-1-1-2"""
def preprocess(path, scale=3):
    """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
    """
    image = imread(path, is_grayscale=True)#!!! always True?
    label_ = modcrop(image, scale)#7-1-1-2-1 crop image for sclaing

    # Must be normalized
    label_ = label_ / 255.

    input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=True)#down-scale
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=True)#up-scale
 
    return input_, label_

"""7-1-1-1 generating image path for training/testing"""

def prepare_data(sess, folderpath):
    """
  Args:
    folderpath: list of path/to/trainfolder and path/to/testfolder
    namelist: list of absolute path of trainImg names and testImg names
    """
    if FLAGS.is_train:#load traning and testing images
        assert(len(folderpath)==2)
        train_filenames = glob.glob(os.path.join(folderpath[0], "*.bmp"))
        test_filenames = glob.glob(os.path.join(folderpath[1], "*.bmp"))
        return [train_filenames,test_filenames]
    else:
        assert(len(folderpath)==1)
        train_filenames = glob.glob(os.path.join(folderpath[0], "*.bmp"))
        return [train_filenames]

"""7-1-1-3"""
def save_each(X,y,path):
    path_x=path+'.X'
    path_y=path+'.y'
    np.save(path_x,X)
    np.save(path_y,y)
    return True
    
def make_data(sess, data, label, folderpath, c_dim,mode='train'):
    print(data.shape)
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if mode=='train':
        savepath = os.path.join(folderpath,'train.c'+str(c_dim)+'.h5')
    elif mode=='test':
        savepath = os.path.join(folderpath, 'test.c'+str(c_dim)+'.h5')
    elif mode=='new':
        savepath = os.path.join(folderpath, 'new.c'+str(c_dim)+'.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        if label is not None:
            hf.create_dataset('label', data=label)

    return True
        
    

def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

"""7-1-1-2-1"""
def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image

def generate_patch(h,w,input_,label_,padding,config):
    #generate patches
    sub_input_sequence=list()
    sub_label_sequence=list()
    nx = 0
    ny = 0 

    for x in range(0, h-config.image_size+1, config.stride):
        nx+=1
        ny=0
        for y in range(0, w-config.image_size+1, config.stride):
            ny+=1
            # We create the inputs and labels.
            # We take care to create all surrounding areas of the patch.

            # left/up
            sub_input1 = input_[x - config.image_size:x, y - config.image_size:y]  # [33 x 33]

            # right/up
            sub_input2 = input_[x + config.image_size:x + 2 * config.image_size, y - config.image_size:y]  # [33 x 33]

            # center/up
            sub_input3 = input_[x:x + config.image_size, y - config.image_size:y]  # [33 x 33]

            # left/center
            sub_input4 = input_[x - config.image_size:x, y:y + config.image_size]  # [33 x 33]

            # center/center
            sub_input5 = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
            sub_label = label_[int(x + padding):int(x + padding + config.label_size), int(y + padding):int(y + padding + config.label_size)]  # [21 x 21]

            # right/center
            sub_input6 = input_[x + config.image_size:x + 2 * config.image_size, y:y + config.image_size]  # [33 x 33]

            # center/bottom
            sub_input7 = input_[x:x + config.image_size, y + config.image_size:y + 2 * config.image_size]  # [33 x 33]

            # left/bottom
            sub_input8 = input_[x - config.image_size:x,y + config.image_size:y + 2 * config.image_size]  # [33 x 33]

            # right/bottom
            sub_input9 = input_[x + config.image_size:x + 2 * config.image_size, y + config.image_size:y + 2 * config.image_size]  # [33 x 33]

            # Make channel value
            # reshape image/label from 2d to 3d
            # Temp array to create higher channel input
            temp_input = np.empty((config.image_size, config.image_size,config.c_dim),dtype=np.float32)
            if(config.c_dim==9):
                listOfInputs = [sub_input1, sub_input2, sub_input3, sub_input4, sub_input5, sub_input6, sub_input7, sub_input8, sub_input9]
            elif(config.c_dim==5):
                listOfInputs = [sub_input3, sub_input4, sub_input5, sub_input6, sub_input7]
            else:#==1
                listOfInputs = [sub_input5]
           
            # nested for loops to stack the high channel inputs
 
            #edge cases
            if ((x -config.image_size)<0 or (x + config.image_size)>0 or (y -config.image_size)<0 or (y + config.image_size)>0):
                for i in range(0, config.c_dim):
                    temp_input[:,:,i] = sub_input5

            #main block
            else:
                for i in range(0, config.c_dim):
                    temp_input[:,:,i] = listOfInputs[i]

            # label is still 1 channel
            sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
            # append to list
            sub_input_sequence.append(temp_input)
            sub_label_sequence.append(sub_label)
    return [nx,ny,sub_input_sequence,sub_label_sequence]
    
"""7-1-1 input setup"""
def input_setup(sess, config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    #if h5 exists, skip
    if not config.make_patch:
        target_path=os.path.join(config.checkpoint_dir,'test.c'+str(config.c_dim)+'.h5')
        if os.path.isfile(target_path):
            return False
            
    # Load data path
    data = prepare_data(sess, [config.trn_folderpath,config.tst_folderpath])#7-1-1-1
    padding =  abs(config.image_size - config.label_size) / 2 # 6
  
    #if training
    trn_sub_input_sequence = []
    trn_sub_label_sequence = []
    tst_sub_input_sequence = []
    tst_sub_label_sequence = []
    #nxny_list=list()
    for i in range(len(data)):
        for j in range(len(data[i])):
            #preprocess each image
            input_, label_ = preprocess(data[i][j], config.scale)#7-1-1-2
            #get image size
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            output=generate_patch(h,w,input_,label_,padding,config)
            if(i==0):#train
                trn_sub_input_sequence.append(output[2])
                trn_sub_label_sequence.append(output[3])
            else:#testing
                tst_sub_input_sequence.append(output[2])
                tst_sub_label_sequence.append(output[3])
                #nxny_list.append((output[0],output[1]))
    #flatten list of lists 
    trn_sub_input_sequence=reduce(lambda x,y: x+y,trn_sub_input_sequence)
    trn_sub_label_sequence=reduce(lambda x,y: x+y,trn_sub_label_sequence)
    tst_sub_input_sequence=reduce(lambda x,y: x+y,tst_sub_input_sequence)
    tst_sub_label_sequence=reduce(lambda x,y: x+y,tst_sub_label_sequence)
    #list to numpy
    X_train=np.asarray(trn_sub_input_sequence)
    y_train=np.asarray(trn_sub_label_sequence)
    X_test=np.asarray(tst_sub_input_sequence)
    y_test=np.asarray(tst_sub_label_sequence)
    
    make_data(sess, X_train, y_train, config.checkpoint_dir, config.c_dim,mode='train')
    make_data(sess, X_test, y_test, config.checkpoint_dir, config.c_dim,mode='test')
    return True

def input_setup_test(sess, config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    #if h5 exists, skip
    if not config.make_patch:
        target_path=os.path.join(config.checkpoint_dir,'new.c'+str(config.c_dim)+'.h5')
        if os.path.isfile(target_path):
            return False
            
    # Load data path
    data = prepare_data(sess, [config.new_image_path])#7-1-1-1
    padding =  abs(config.image_size - config.label_size) / 2 # 6
  
    #if training
    tst_sub_input_sequence = []
    tst_sub_label_sequence = []
    nxny_list=list()
    for j in range(len(data[0])):
        #preprocess each image
        input_, label_ = preprocess(data[0][j], config.scale)#7-1-1-2
        #get image size
        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        output=generate_patch(h,w,input_,label_,padding,config)

        tst_sub_input_sequence.append(output[2])
        tst_sub_label_sequence.append(output[3])
        nxny_list.append((output[0],output[1]))
    #flatten list of lists 
    tst_sub_input_sequence=reduce(lambda x,y: x+y,tst_sub_input_sequence)
    #tst_sub_label_sequence=reduce(lambda x,y: x+y,tst_sub_label_sequence)
    #list to numpy
    X_test=np.asarray(tst_sub_input_sequence)
    #y_test=np.asarray(tst_sub_label_sequence)
    
    make_data(sess, X_test,None, config.checkpoint_dir, config.c_dim,mode='new')
    return nxny_list,data[0]
    
def imsave(image, path):
    return scipy.misc.imsave(path, image,format='bmp')
#"""7-1-2 merge patches into an image"""
def merge(patches, nxny):
    patches=np.asarray(patches)
    print(patches.shape)
    h, w = patches.shape[1], patches.shape[2]
    img = np.zeros((h*nxny[0], w*nxny[1],1))
    for idx, image in enumerate(patches):
        #print(image.shape)
        i = idx % nxny[1]
        j = idx // nxny[1]
        img[j*h:j*h+h, i*w:i*w+w,:] = image
    
    return np.squeeze(img)