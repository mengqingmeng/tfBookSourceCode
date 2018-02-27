import tensorflow as tf
#线性回归模型
class LineRegModel:
    def __init__(self):
        self.a_val=tf.Variable(tf.random_normal([1]))#权重，正态分布
        self.b_val = tf.Variable(tf.random_normal([1]))#偏置
        self.x_input = tf.placeholder(tf.float32)   #输入占位
        self.y_label = tf.placeholder(tf.float32)   #标签占位
        self.y_output = tf.add(tf.multiply(self.x_input,self.a_val),self.b_val) #模型计算值
        self.loss = tf.reduce_mean(tf.pow(self.y_output-self.y_label,2))

    def get_op(self):
        return tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

    def get_saver(self):
        return tf.train.Saver()