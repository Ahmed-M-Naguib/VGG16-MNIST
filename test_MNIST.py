
import tensorflow as tf
from VGG19_MNIST_Model import Vgg19
from load_MNIST import mnist
from scipy import misc



trX, _, trY, _ = mnist(data_dir = 'mnist/data/')

trX=trX[0:100]
trY=trY[0:100]

vgg = Vgg19(vgg19_npy_path=None,
             dropout=0.5,
             batch_size=32,
             lr_vgg=0.0015,
             beta1 = 0.5,
             logs_dir = "logs",
             nClasses = 10,
             fixed_layers={},
             delete_layers={})



