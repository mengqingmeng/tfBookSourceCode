import cv2
import os
import numpy as np
import tensorflow as tf
from skimage import io


#数据集加工，将图片改为227大小，存起来
def rebuild(dir):
    for root,dirs,files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root,file)
            try:
                image = cv2.imread(filepath)
                dim = (227,227)
                resized = cv2.resize(image,dim)
                path = "E:\\Workspace\\DL\\DATAS\\cat_vs_dog\\train\\train"
                cv2.imwrite(path,resized)
            except:
                print(filepath)
                os.remove(filepath)
    cv2.waitKey(0)

#将图片数据转为tensorflow专用格式
def get_file(file_dir):
    images = []
    temp = []
    for root,sub_folders,files in os.walk(file_dir):
        #图片目录
        for name in files:
            images.append(os.path.join(root,name))
        for name in sub_folders:
            temp.append(os.path.join(root,name))

        print(files)
    labels=[]

    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('\\')[-1]
        if letter =='cat':
            labels = np.append(labels,n_img*[0])
        else:
            labels = np.append(labels,n_img*[1])

    temp = np.array([images,labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]

    return image_list,label_list

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def convert_to_tfrecord(image_list,label_list,save_dir,name):
    filename = os.path.join(save_dir,name+'.tfrecords')
    n_samples = len(label_list)
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start ...')
    for i in np.arange(0,n_samples):
        try:
            image = io.imread(image_list[i])
            image_raw = image.tostring()
            label = int(label_list[i])
            example = tf.train.Example(features=tf.train.Features(feature={'label':int64_feature(label),
                                                                           'image_raw':bytes_feature(image_raw)}))
            writer.write(example.SerializerToString())
        except IOError as e:
            print('Could not read:',image_list[i])

        writer.close()
        print('Transform done!')



def read_and_decode(tfrecords_file,batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label':tf.FixedLenFeature([],tf.int64),
            'image_raw':tf.FixedLenFeature([],tf.string),
            })
    image = tf.deocde_raw(img_features['image_raw'],tf.uint8)
    image = tf.reshape(image,[227,227,3])
    label = tf.cast(img_features['label'],tf.int32)
    image_batch,label_batch=tf.train.shuffle_batch([image,label],
                                                   batch_size=batch_size,
                                                   min_after_dequeue=100,
                                                   num_threads=64,
                                                   capacity=200)
    return image_batch,tf.reshape(label_batch,[batch_size])

def get_batch(image_list,label_list,img_width,img_height,batch_size,capacity):
    image = tf.cast(image_list,tf.string)
    label = tf.cast(label_list,tf.int32)

    input_queue=tf.train.slice_input_producer([image,label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
    image = tf.image.per_image_standardization(image)
    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)
    label_batch=tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch

def one_hot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample),labels]=1
    return onehot_labels

#img_list,label_list = get_file("E:\\Workspace\\DL\\DATAS\\cat_vs_dog\\train\\train")
#get_batch(img_list,label_list,224,224,120,120)