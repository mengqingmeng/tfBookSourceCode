import tensorflow as tf

#重新定义线性回归模型
#使用tf.variable_scope("var")对变量的域进行了定义，每个变量或者占位符定义了名称
class LineRegModel:

    def __init__(self):
        with tf.variable_scope("var"):
            self.a_val = tf.Variable(tf.random_normal([1]),name="a_val")
            self.b_val = tf.Variable(tf.random_normal([1]),name="b_val")
        self.x_input = tf.placeholder(tf.float32,name="input_placeholder")
        self.y_label = tf.placeholder(tf.float32,name="result_placeholder")
        self.y_output = tf.add(tf.multiply(self.x_input, self.a_val), self.b_val,name="output")
        self.loss = tf.reduce_mean(tf.pow(self.y_output - self.y_label, 2))

    def get_saver(self):
        return tf.train.Saver()

    def get_op(self):
        return tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
