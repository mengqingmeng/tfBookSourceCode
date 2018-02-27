import tensorflow as tf
import save_and_restore.global_variable as global_variable
from save_and_restore import lineRegulation_model as model
import time

#恢复模型并预测
begin = time.time();
model = model.LineRegModel()
x_input = model.x_input
y_output = model.y_output

saver = model.get_saver()
sess = tf.Session()
saver.restore(sess,global_variable.save_path)

result = sess.run(y_output,feed_dict={x_input:[2]})
end = time.time();
print(result)
print('cost:',(end-begin)/1000.0)
