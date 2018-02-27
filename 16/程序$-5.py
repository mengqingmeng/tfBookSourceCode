import tensorflow as tf
import numpy as np
import save_and_restore.global_variable as global_variable
from save_and_restore import lineRegulation_model_v2 as model

train_x = np.random.rand(5)
train_y = 5 * train_x + 3.2   # y = 5 * x + 3
model = model.LineRegModel()

a_val = model.a_val
b_val = model.b_val

x_input = model.x_input
y_label = model.y_label

y_output = model.y_output

loss = model.loss
optimize = model.get_op()
saver = model.get_saver()

if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    flag = True
    epoch = 0
    while flag:
        epoch += 1
        _ , loss_val = sess.run([optimize,loss],feed_dict={x_input:train_x,y_label:train_y})
        if loss_val < 1e-6:
            flag = False
    print(a_val.eval(sess) , "   ", b_val.eval(sess))
    print("-----------%d-----------"%epoch)
    print(a_val.op)#打印节点内容
    savePath = global_variable.save_path+"save_model"
    saver.save(sess,savePath)
    print("model save finished")
    sess.close()
