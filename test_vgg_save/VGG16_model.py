import numpy as np
import tensorflow as tf
import save_and_restore.global_variable as global_variable
import vgg16_weights_and_class

class vgg16:
    def __init__(self, imgs):
        self.parameters = []
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = self.fc8

    def saver(self):
        return tf.train.Saver()

    def maxpool(self,name,input_data,trainable=True):
        """
        池化
        :param name:
        :param input_data:
        :param trainable:
        :return:
        """
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out

    def conv(self,name, input_data, out_channel,trainable=True):
        """
        定义卷基
        :param name:   域名
        :param input_data:输入数据
        :param out_channel:输出通道数，就是卷积核的个数
        :param trainable: 参数是否可训练
        :return:
        """
        in_channel = input_data.get_shape()[-1] #图片通道数
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32,trainable=False) #卷积核
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32,trainable=False) #偏置
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")   #卷积
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]
        return out

    def fc(self,name,input_data,out_channel,trainable = True):
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32,trainable = trainable)
            biases = tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32,trainable = trainable)
            res = tf.matmul(input_data_flat,weights)
            out = tf.nn.relu(tf.nn.bias_add(res,biases))
        self.parameters += [weights, biases]
        return out

    def convlayers(self):
        """
        定义vgg模型，采用书中配置c
        :return:
        """
        #conv1
        self.conv1_1 = self.conv("conv1re_1",self.imgs,64,trainable=False)
        self.conv1_2 = self.conv("conv1_2",self.conv1_1,64,trainable=False)
        self.pool1 = self.maxpool("poolre1",self.conv1_2,trainable=False)

        #conv2
        self.conv2_1 = self.conv("conv2_1",self.pool1,128,trainable=False)
        self.conv2_2 = self.conv("convwe2_2",self.conv2_1,128,trainable=False)
        self.pool2 = self.maxpool("pool2",self.conv2_2,trainable=False)

        #conv3
        self.conv3_1 = self.conv("conv3_1",self.pool2,256,trainable=False)
        self.conv3_2 = self.conv("convrwe3_2",self.conv3_1,256,trainable=False)
        self.conv3_3 = self.conv("convrew3_3",self.conv3_2,256,trainable=False)
        self.pool3 = self.maxpool("poolre3",self.conv3_3,trainable=False)

        #conv4
        self.conv4_1 = self.conv("conv4_1",self.pool3,512,trainable=False)
        self.conv4_2 = self.conv("convrwe4_2",self.conv4_1,512,trainable=False)
        self.conv4_3 = self.conv("conv4rwe_3",self.conv4_2,512,trainable=False)
        self.pool4 = self.maxpool("pool4",self.conv4_3,trainable=False)


        #conv5
        self.conv5_1 = self.conv("conv5_1",self.pool4,512,trainable=False)
        self.conv5_2 = self.conv("convrwe5_2",self.conv5_1,512,trainable=False)
        self.conv5_3 = self.conv("conv5_3",self.conv5_2,512,trainable=False)
        self.pool5 = self.maxpool("poorwel5",self.conv5_3,trainable=False)

    def fc_layers(self):
        self.fc6 = self.fc("fc6", self.pool5, 4096,trainable=False)
        self.fc7 = self.fc("fc7", self.fc6, 4096,trainable=False)
        self.fc8 = self.fc("fc8", self.fc7, 2)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i not in [30,31]:
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----------all done---------------")

def onehot(labels):
    n_sample = len(labels)
    n_class=max(labels) + 1
    onehot_labels = np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample),labels] = 1
    return onehot_labels