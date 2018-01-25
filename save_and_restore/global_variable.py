import os
d = os.path.dirname(__file__)
save_path = os.path.join(d,'model/')
if not os.path.exists(save_path):
    os.mkdir(save_path)