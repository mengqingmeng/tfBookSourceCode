import os
d = os.path.dirname(__file__)
#保存模型在当前文件夹下面的model文件夹中
save_path = os.path.join(d,'model\\')
if not os.path.exists(save_path):
    os.mkdir(save_path)

flowers_data_dir="..\\flowers"

save_model = os.path.join(d,'model\\')

pre_ckpt_save_model= os.path.join(d,"inception_resnet_v2_2016_08_30\\")

logdir_path = os.path.join(d,"log_dir")