import tensorflow as tf
from scipy.misc import imread,imresize
from test_vgg_save import VGG16_model as model
import os
import numpy as np

imgs = tf.placeholder(tf.float32,[None,224,224,3])
sess =tf.Session()
vgg = model.vgg16(imgs)
fc3_cat_and_dog=vgg.probs
saver = vgg.saver()
saver.restore(sess,'..\\vgg_finetuning_model\\')

# for root,sub_floders,files in os.walk('E:\\Workspace\\DL\\DATAS\\cat_vs_dog\\test1\\test1'):
#     i = 0
#     cat = 0
#     dog = 0
#     for name in files:
#         i+=1
#         filepath = os.path.join(root,name)
#
#         try:
#             img1 = imread(filepath,mode='RGB')
#             img1 = imresize(img1,(224,224))
#             probs = sess.run(fc3_cat_and_dog, feed_dict={vgg.imgs: [img1]})
#             prob = probs[0]
#             max_index = np.argmax(prob)
#             if max_index == 0:
#                 cat += 1
#             else:
#                 dog += 1
#             if i % 50 == 0:
#                 acc = (cat * 1.) / (dog+cat)
#                 print(acc)
#                 print('---img number is %d---' % i)
#         except:
#             print('remove:',filepath)
classnams=['cat','dog']
img1 = imread("E:\\Workspace\\DL\\DATAS\\cat_vs_dog\\train\\train\\cat\\cat.4.jpg",mode='RGB')
img1 = imresize(img1,(224,224))


prob = sess.run(fc3_cat_and_dog, feed_dict={vgg.imgs: [img1]})[0]
preds = (np.argsort(prob)[::-1])[0:2]
for p in preds:
    print(classnams[p], prob[p])