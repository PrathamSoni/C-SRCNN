#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate batch inx to slice numpy in memory
"""
import numpy as np
import math
class dataLoader(object):
    def __init__(self,
                 dataSize,
                 batchSize=128,
                 shuffle=True,
                 seed=0):
        self.dataSize=dataSize
        self.seq=np.arange(self.dataSize)
        self.batchSize=batchSize
        self.shuffle=shuffle
        if(self.shuffle):
            np.random.seed(seed)
        self.pointer=0
        self.pointer_upperbound=math.ceil(self.dataSize/self.batchSize)
    def shuffleSeq_(self):
        if self.shuffle:
            self.seq=np.random.permutation(self.dataSize)
            return True
        else:
            #print('This is a FIFO data loader. You cannot shuffle it.')
            return False
    def get_batch(self):
        if self.pointer>=self.pointer_upperbound:
            self.pointer=0
        if self.pointer==0:
            self.shuffleSeq_()
        inx=self.seq[self.pointer*self.batchSize:(self.pointer+1)*self.batchSize].view()
        self.pointer+=1
        return inx
            
        
       
