import tensorflow as tf
import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    def __init__(self,
                 vgg19_npy_path=None,
                 dropout=0.5,
                 batch_size=32,
                 lr_vgg=0.0015,
                 beta1 = 0.5,
                 logs_dir = "logs",
                 nClasses = 10,
                 fixed_layers={},
                 delete_layers={}):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None
        self.var_dict = {}
        self.dropout = dropout
        self.logs_dir = logs_dir
        self.beta1 = beta1
        self.lr_vgg = lr_vgg
        self.batch_size = batch_size
        self.nClasses = nClasses
        self.build(fixed_layers, delete_layers)
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

    def build(self, fixed_layers={}, delete_layers={}):
        self.labels = tf.placeholder(tf.int32, shape=[self.batch_size, self.nClasses], name='labels')
        self.inputs2D = tf.placeholder(tf.float32, shape=[self.batch_size, 224, 224, 3], name='input2D')

        rgb_scaled = self.inputs2D * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1", not "conv1_1" in fixed_layers)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", not "conv1_2" in fixed_layers)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1", not "conv2_1" in fixed_layers)
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", not "conv2_2" in fixed_layers)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1", not "conv3_1" in fixed_layers)
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", not "conv3_2" in fixed_layers)
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3", not "conv3_3" in fixed_layers)
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4", not "conv3_4" in fixed_layers)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1", not "conv4_1" in fixed_layers)
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2", not "conv4_2" in fixed_layers)
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3", not "conv4_3" in fixed_layers)
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4", not "conv4_4" in fixed_layers)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1", not "conv5_1" in fixed_layers)
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2", not "conv5_2" in fixed_layers)
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3", not "conv5_3" in fixed_layers)
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4", not "conv5_4" in fixed_layers)
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6", not "fc6" in fixed_layers)  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if(not "fc6" in fixed_layers):
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7", not "fc7" in fixed_layers)
        self.relu7 = tf.nn.relu(self.fc7)
        if (not "fc7" in fixed_layers):
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8", not "fc8" in fixed_layers)
        self.relu8 = tf.nn.relu(self.fc8)
        if (not "fc8" in fixed_layers):
            self.relu8 = tf.nn.dropout(self.relu8, self.dropout)

        self.fc9 = self.fc_layer(self.relu8, 1000, self.nClasses, "fc9", not "fc9" in fixed_layers)
        self.prob = tf.nn.softmax(self.fc9, name="prob")

        self.data_dict = None


    def convert_MNIST_labels(self, labels_MNIST):
        labels=[]
        return labels
    def convert_MNIST_images(self, images_MNIST):
        images=[]
        return images

    def test(self, images):
        return self.sess.run(self.prob, feed_dict={self.inputs2D: images})
    def train(self, images, labels, epochs=500):
        saver = tf.train.Saver()
        self.vgg_loss = tf.reduce_sum((self.prob - self.labels) ** 2)
        self.vgg_optim = tf.train.AdamOptimizer(self.lr_vgg, beta1=self.beta1).minimize(self.vgg_loss)
        self.vgg_loss_sum = tf.summary.scalar("vgg_loss", self.vgg_loss)
        self.data_size = images.get_shape().as_list()[0,]
        self.train_writer = tf.summary.FileWriter("./logs", graph_def=self.sess.graph_def)
        step = 0
        for epoch in range (epochs):
            print('epoch: %d' % epoch)
            batch_idxes = self.data_size// self.batch_size
            for batch_idxs in range (batch_idxes):
                step =step +1
                batch_2D = images[batch_idxs*self.batch_size:batch_idxs*self.batch_size+self.batch_size]
                labels   = labels[batch_idxs * self.batch_size:batch_idxs * self.batch_size + self.batch_size]
                _, loss = self.sess.run([self.vgg_optim, self.vgg_loss_sum], feed_dict={self.inputs2D: batch_2D, self.labels: labels})
                #if batch_idxs == 0:
                #    sample,g_loss = sess.run([self.prob,self.vgg_loss],feed_dict={self.inputs2D: batch_2D, self.labels: labels})
                #    print('output_onelayer',sample)
                self.train_writer.add_summary(loss,step)
        saver.save(self.sess, "save/model.ckpt")
        print('saved ...\n')
    def test_MNIST(self, images):
        self.test(self.convert_MNIST_images(images))
    def train_MNIST(self, images, labels, epochs=500):
        self.train(self.convert_MNIST_images(images), self.convert_MNIST_labels(labels), epochs)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    def conv_layer(self, bottom, in_channels, out_channels, name, trainable):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, trainable)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu
    def fc_layer(self, bottom, in_size, out_size, name, trainable):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, trainable)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
    def get_conv_var(self, filter_size, in_channels, out_channels, name, trainable):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters",trainable)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases",trainable)

        return filters, biases
    def get_fc_var(self, in_size, out_size, name, trainable):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights", trainable)

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)

        return weights, biases
    def get_var(self, initial_value, name, idx, var_name, trainable):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var
    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path
    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
