import tensorflow as tf
import numpy as np
#import save_and_restore.global_variable as global_variable
#from save_and_restore import lineRegulation_model as model
import lineModel as model
#训练线性回归模型，并保存模型

#数据集
train_x = np.random.rand(5) #输入
train_y = 5 * train_x + 3.2  #标签 y = 5 * x + 3

#模型
model = model.LineRegModel()

#参数
a_val = model.a_val #权重
b_val = model.b_val #偏置

#输入数据
x_input = model.x_input
y_label = model.y_label

#y_output = model.y_output

#损失函数
loss = model.loss

optimize = model.get_op()
saver = tf.train.Saver()
if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())#初始化变量
    flag = True
    epoch = 0
    while flag:
        epoch += 1
        _ , loss_val = sess.run([optimize,loss],feed_dict={x_input:train_x,y_label:train_y}) #给输入数据赋值，并运行
        if loss_val < 1e-6:
            flag = False
    #print(a_val.eval(sess) , "   ", b_val.eval(sess))
    #输出权重和偏置
    print(sess.run([a_val,b_val]))
    print("-----------%d-----------"%epoch)

    #保存模型
    saver.save(sess,'model/')
    print("model save finished")
    sess.close()
