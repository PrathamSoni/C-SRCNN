from utils import (
  read_data, 
  input_setup, 
  input_setup_test,
  imsave,
  preprocess,
  merge
)
import glob
import numpy as np
import gc
from functools import reduce
import math
import time
import os
import tensorflow as tf
from dataLoader import dataLoader


class SRCNN(object):
    """6-1 init SRCNN and setup hyperparameters"""
    def __init__(self, 
               sess, 
               config):

        self.sess = sess
        self.config=config
        self.build_model()
        
    """6-2 define model"""
    def build_model(self):
        #input
        self.images = tf.placeholder(tf.float32, [None, self.config.image_size, self.config.image_size, self.config.c_dim], name='images')
        #output
        self.labels = tf.placeholder(tf.float32, [None, self.config.label_size, self.config.label_size, 1], name='labels')
        #weights
        self.weights = {
          'w1': tf.Variable(tf.truncated_normal([9, 9, self.config.c_dim, 64], stddev=1e-3, seed=111),name='w1'),
          'w2': tf.Variable(tf.truncated_normal([5, 5, 64, 32], stddev=1e-3, seed=222),name='w2'),
          'w3': tf.Variable(tf.truncated_normal([5, 5, 32, 1], stddev=1e-3, seed=333),name='w3'),
          }
        #bias
        self.biases = {
          'b1': tf.Variable(tf.constant(0.1,shape=[64]), name='b1'),
          'b2': tf.Variable(tf.constant(0.1,shape=[32]), name='b2'),
          'b3': tf.Variable(tf.constant(0.1,shape=[1]), name='b3'),
          }
        #prediction
        self.pred = self.model()
        # Loss function (MSE) #avg per sample
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        #to save best model
        self.saver = tf.train.Saver()
        
    """7-1 train/test"""
    def input_parser(self,img_path):
        img,lbl=preprocess(img_path)
        img=np.asarray([img]*self.config.c_dim).astype(np.float32)
        img=np.transpose(img,(1,2,0))#channel at tail
        return img,lbl
    def test(self):
        #load new images in a folder
        try:
            self.load(self.config.checkpoint_dir)
            print(" [*] Load SUCCESS")
        except:
            print(" [!] Load failed...")
            return

        print('new_data_folder',self.config.new_image_path)

        nxny_list,namelist=input_setup_test(self.sess,self.config)
        new_data_dir = os.path.join(self.config.checkpoint_dir,'new.c'+str(self.config.c_dim)+'.h5')
        X_test,_=read_data(new_data_dir)
        tst_data_loader=dataLoader(dataSize=X_test.shape[0],
                                   batchSize=self.config.test_batch_size,
                                   shuffle=False)
        tst_batch_count=int(math.ceil(X_test.shape[0]/self.config.test_batch_size))
        #print(X_test[0].shape)
        #print(X_test[1].shape)
        #new_data_loader=tf.data.Dataset.from_tensor_slices(X_test)
        #new_data_loader = new_data_loader.batch(batch_size=self.config.test_batch_size)
        #iterator = tf.data.Iterator.from_structure(new_data_loader.output_types,new_data_loader.output_shapes)
        #next_batch=iterator.get_next()
        #new_init_op = iterator.make_initializer(new_data_loader)
        
        result=list()
        #self.sess.run(new_init_op)
        start_time=time.time()
        for batch in range(0,tst_batch_count):
            inx=tst_data_loader.get_batch()
            X=X_test[inx].view()#self.sess.run(next_batch)
            y_pred = self.pred.eval({self.images: X})
            result.append(y_pred)
                #total_mse+=tf.reduce_mean(tf.squared_difference(y_pred, y))
                #batch_count+=1

        #averge_mse=total_mse/batch_count
        #PSNR=-10*math.log10(averge_mse)
        print("time: [%4.2f]" % (time.time()-start_time))
        
        #save
            #flatten
        print(len(result))
        output=list()
        for i in result:
            for j in range(0,i.shape[0]):
                output.append(i[j])
        print(len(output))

        
        #result=[self.sess.run(i) for i in result]
        patch_inx=0
        for i in range(0,len(nxny_list)):
            nx,ny=nxny_list[i]
            img=merge(output[patch_inx:(patch_inx+nx*ny)],(nx,ny))
            print('img shape@',i,img.shape)
            patch_inx+=nx*ny
            imsave(img,namelist[i].replace('.bmp','.bmp.c'+str(self.config.c_dim)))
                    
    def train(self):
        #data preprocessing
        if(input_setup(self.sess, self.config)):#7-1-1
            print('generating patches...')
        else:
            print('found existing h5 files...')

        #build image path  
        trn_data_dir = os.path.join(self.config.checkpoint_dir,'train.c'+str(self.config.c_dim)+'.h5')
        print('trn_data_dir',trn_data_dir)
        X_train,y_train=read_data(trn_data_dir)
        trn_data_loader=dataLoader(dataSize=X_train.shape[0],
                                   batchSize=self.config.batch_size,
                                   shuffle=True,
                                   seed=123)
        
        tst_data_dir = os.path.join(self.config.checkpoint_dir,'test.c'+str(self.config.c_dim)+'.h5')
        print('tst_data_dir',tst_data_dir)
        X_test,y_test=read_data(tst_data_dir)#7-1-2 read image from h5py
        tst_data_loader=dataLoader(dataSize=X_test.shape[0],
                                   batchSize=self.config.test_batch_size,
                                   shuffle=False)
        
        #data description
        print('X_train.shape',X_train.shape)
        print('y_train.shape',y_train.shape)
        print('X_test.shape',X_test.shape)
        print('y_test.shape',y_test.shape)
        #del X_train,y_train,X_test,y_test
        #gc.collect()

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
  
        tf.global_variables_initializer().run()###remove DEPRECATED function###tf.initialize_all_variables().run()
    
        #Try to load pretrained model from checkpoint_dir
        if self.load(self.config.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        #if training
        print("Training...")
        batch_count=int(math.ceil(X_train.shape[0]/self.config.batch_size))
        tst_batch_count=int(math.ceil(X_test.shape[0]/self.config.test_batch_size))
        best_PSNR=0.
        best_ep=0.
        patience=self.config.patience
        trn_PSNR_record=list()
        trn_loss_record=list()
        tst_PSNR_record=list()
        tst_loss_record=list()
        training_loss=tf.placeholder(tf.float32)

        training_PSNR=tf.placeholder(tf.float32)

        validation_loss=tf.placeholder(tf.float32)

        validation_PSNR=tf.placeholder(tf.float32)
        

        tf.summary.scalar('Training MSE', training_loss)
        tf.summary.scalar('Training PSNR', training_PSNR)
        tf.summary.scalar('Validation MSE',validation_loss)
        tf.summary.scalar('Validation PSNR', validation_PSNR)
        tf.summary.histogram('w1',self.weights['w1'])
        tf.summary.histogram('b1',self.biases['b1'])
        tf.summary.histogram('w2',self.weights['w2'])
        tf.summary.histogram('b2',self.biases['b2'])
        tf.summary.histogram('w3',self.weights['w3'])
        tf.summary.histogram('b3',self.biases['b3'])
        tf.summary.image('images', tf.convert_to_tensor(self.images[:,:,:,(self.config.c_dim-1)//2:(self.config.c_dim+1)//2]))
        tf.summary.image('labels', tf.convert_to_tensor(self.labels))
        tf.summary.image('predicted',tf.convert_to_tensor(self.pred))
        merged=tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.config.summary_dir + '/train', self.sess.graph)

        tf.global_variables_initializer().run()###remove DEPRECATED function###tf.initialize_all_variables().run()
        for ep in range(self.config.epoch):#for each epoch
            epoch_loss = 0.
            average_loss1 = 0.
            start_time = time.time()
            for batch in range(batch_count):
                inx=trn_data_loader.get_batch()
                X,y = X_train[inx],y_train[inx]
                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: X, self.labels: y})#update weights and biases 
                    #print('err',err)
                epoch_loss += err            
            average_loss1 = epoch_loss / batch_count #per sample
            average_loss1=average_loss1.astype(np.float32)
            trn_loss_record.append(average_loss1)
            #print(self.sess.run(average_loss1))
            PSNR1=-10*math.log10(average_loss1)
            trn_PSNR_record.append(PSNR1)
            print("Epoch: [%2d], \n\ttime: [%4.2f], \n\ttraining loss: [%.8f], \n\tPSNR: [%.4f]" % (ep, time.time()-start_time, average_loss1,PSNR1))
            
            #valid
            epoch_loss = 0.
            average_loss = 0.
            start_time = time.time()
            for batch in range(tst_batch_count):
                inx=tst_data_loader.get_batch()
                X,y = X_test[inx],y_test[inx]
                err = self.sess.run(self.loss, feed_dict={self.images: X, self.labels: y})#only compute err
                epoch_loss += err
            average_loss = epoch_loss / tst_batch_count #per sample
            average_loss=average_loss.astype(np.float32)
            tst_loss_record.append(average_loss)
            PSNR=-10*math.log10(average_loss) 
            tst_PSNR_record.append(PSNR)
            print("\n\ttime: [%4.2f], \n\ttesting loss: [%.8f], \n\tPSNR: [%.4f]\n\n" % (time.time()-start_time, average_loss,PSNR))

            summary=self.sess.run(merged, feed_dict={training_loss: average_loss1, training_PSNR: PSNR1, validation_loss:average_loss, validation_PSNR:PSNR,self.images: X, self.labels: y})

            train_writer.add_summary(summary,ep)
            print("added")
            
            
            #save
            if PSNR<=best_PSNR:
                patience-=1
                if patience==0:
                    print('early stop!')
                    break
            else:# PSNR>best_PSNR:
                #print('\tcurrent best PSNR: <%.4f>\n' % PSNR)
                self.save(self.config.checkpoint_dir,ep)
                best_ep=ep
                best_PSNR=PSNR
                patience=self.config.patience
        print('best ep',best_ep)
        print('best PSNR',best_PSNR)
        #save
        info=np.vstack((np.asarray(trn_loss_record),np.asarray(trn_PSNR_record),np.asarray(tst_loss_record),np.asarray(tst_PSNR_record)))
        np.save(os.path.join(self.config.checkpoint_dir,'info'),info)
        print('info saved!',info.shape)

    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3']
        #out = tf.clip_by_value(conv3,0.0,1.0)
        return conv3#out

    def save(self, checkpoint_dir, step):
        model_name = "CASRCNN_C"+str(self.config.c_dim)+".model"
        model_dir = "%s_%s_%s" % ("srcnn", self.config.label_size, self.config.c_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s_%s" % ("srcnn", self.config.model_label_size, self.config.c_dim)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print('checkpoint_dir',checkpoint_dir)#print folder path out
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('model_checkpoint_path',ckpt.model_checkpoint_path)#model path
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
