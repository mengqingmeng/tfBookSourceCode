import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
from vgg16_weights_and_class.imagenet_classes import class_names
import VGG16_model as model
if __name__ == '__main__':
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = model.vgg16(imgs)
    prob = vgg.probs

    sess = tf.Session()
    vgg.load_weights("..\\vgg16_weights_and_class\\vgg16_weights.npz",sess)

    img1 = imread('..\\img\\dog.jpg', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(prob, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])
