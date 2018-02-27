import tensorflow as tf
import save_and_restore.global_variable as global_variable
savePath = global_variable.save_path
saver = tf.train.import_meta_graph(savePath+'save_model.meta')#恢复图文件
#model = tf.train.latest_checkpoint(savePath)
saver = tf.train.Saver()
#读取placeholder和最终的输出结果
graph = tf.get_default_graph()
a_val = graph.get_tensor_by_name('var/a_val:0')

input_placeholder=graph.get_tensor_by_name('input_placeholder:0')
labels_placeholder=graph.get_tensor_by_name('result_placeholder:0')
y_output=graph.get_tensor_by_name('output:0')#最终输出结果的tensor


with tf.Session() as sess:
    saver.restore(sess, savePath+'save_model.data-00000-of-00001')#恢复权值
    #saver.restore(sess,model)
    result = sess.run(y_output, feed_dict={input_placeholder: [1]})
    print(result)
    print(sess.run(a_val))
