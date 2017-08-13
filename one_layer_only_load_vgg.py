
import tensorflow as tf
# import ops
import math
import os
from PIL import Image
import glob
import numpy as np
import scipy.io as sc
# import cv2 as cv2
import utils
import vgg19_trainable as vgg19

from scipy import misc

from visualize import *
class gan():
    def __init__(self,batch_size=2,
                 image_path='./test_data',
                 data_size=2,
                 lr_G=0.0015,
                 beta1 = 0.5,
                 logs_dir = "logs"):


        self.logs_dir = logs_dir
        self.beta1 = beta1
        self.lr_G = lr_G
        self.data_size=data_size
        self.batch_size = batch_size
        self.image_path = image_path
        self.load_examples()
        self.create_model()


    def test(self):
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "save/model.ckpt")
            print("Loaded saved session.\n")
            self.load_examples()
            idxs = len(self.images2D) // self.batch_size
            print(idxs)

            for batch_idxs in range(idxs):
                test_data = self.images2D[batch_idxs * self.batch_size:batch_idxs * self.batch_size + self.batch_size]
                sample= sess.run(self.G,feed_dict={ self.inputs2D: test_data})
                d=sample*1
                for i in range(len(d)):
                    misc.imsave( "./out/sample_%04d.jpg" % (batch_idxs * self.batch_size + i),d[i,:,:,:])


    def train(self):
        #tensorboard --logdir=logs
        epochs = 500
        if tf.gfile.Exists(self.logs_dir):
            tf.gfile.DeleteRecursively(self.logs_dir)
        tf.gfile.MakeDirs(self.logs_dir)

        for var in tf.trainable_variables():
            if var.name.startswith("generator"):
                print ("g"+var.name)
        print('end')
        # self.vgg_vars=[var for  var in tf.trainable_variables() if var.name.startswith("vgg")]
        # vgg_optim = tf.train.AdamOptimizer(self.lr_G, beta1=self.beta1) \
        #     .minimize(self.vgg_loss, var_list=self.vgg_vars)

        self.g_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        g_optim = tf.train.AdamOptimizer(self.lr_G, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        saver = tf.train.Saver()
        self.g_sum = tf.summary.merge([self.g_loss_sum])
        step = 0

        with tf.Session() as sess:

            self.train_writer = tf.summary.FileWriter("./logs", graph_def=sess.graph_def)
            sess.run(tf.global_variables_initializer())

            for epoch in range (epochs):
                print('epoch: %d' % epoch)
                batch_idxs = self.data_size// self.batch_size
                print(batch_idxs)
                for batch_idxs in range (batch_idxs):
                    step =step +1
                    batch_2D = self.images2D[batch_idxs*self.batch_size:batch_idxs*self.batch_size+self.batch_size]

                    # results_vgg = sess.run(vgg_optim, feed_dict={self.inputs2D: batch_2D, self.train_mode: True})
                    prob = sess.run(self.vgg.prob, feed_dict={self.inputs2D: batch_2D, self.train_mode: False})
                    _,results_g = sess.run([g_optim,self.g_sum],feed_dict={self.input_onelayers :prob})

                    if np.mod(step,1) == 0:
                        sample,g_loss = sess.run([self.G,self.g_loss],
                                                        feed_dict={self.input_onelayers :prob})
                        print('output_onelayer',sample)
                    print('epoch: %d'% epoch)

                    self.train_writer.add_summary(results_g,step)
            saver.save(sess, "save/model.ckpt")
            print('saved ...\n')

    def create_model(self):

        self.train_mode = tf.placeholder(tf.bool)
        self.inputs2D = tf.placeholder(tf.float32,shape = [self.batch_size,224,224,3],name='input2D')
        self.input_onelayers = tf.placeholder(tf.float32,shape = [self.batch_size,1000],name='input_onelayer')


        with tf.variable_scope("generator") as scope:
            self.G= self.generator(self.input_onelayers)

        self.vgg = vgg19.Vgg19('./vgg19.npy')
        self.vgg.build(self.inputs2D, self.train_mode)

        cluster2 = tf.cast(self.clusters, tf.float32)
        self.cluster_L=tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=cluster2,logits=self.probability_enc),reduction_indices=1))
        self.g_loss = self.cluster_L
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        # self.vgg_loss=tf.reduce_sum((self.vgg.prob-self.datalabel)**2)

    def generator(self,input2D):
        layers = []
        #  Fully connected layer
        with tf.variable_scope("Fully_Connected_enc"):
            dim=input2D.get_shape()[1].value
            weight=tf.get_variable("filter",[dim,2],dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.02))
            linear=tf.matmul(input2D,weight)
            biases=tf.get_variable("b",2,dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            self.probability_enc=tf.nn.bias_add(linear,biases)
            self.clusters=tf.sigmoid(self.probability_enc)>0.5
            layers.append(self.probability_enc)
        return layers[-1]

    def load_examples(self):
        images=np.zeros([self.data_size,224,224,3])

        for i in range(1,len(images),1):
            imgPath = self.image_path + '/a (' + str(i) + ').jpeg'
            print(imgPath)
            img=misc.imread(imgPath)
            image = misc.imresize(img, [224, 224])
            images[i-1, :, :, :] = image
        self.images2D = images.astype(np.float32)/255
        self.datalabel=np.array(([[1,0],[0,1]]))

def main():
    model = gan()

    my_file = Path("save/model.ckpt.index")
    if my_file.exists()==False:
        print("file couldn't be found.\nTraining from the beginning ...\n")
        model.train()
    else:
        model.test()

main()



