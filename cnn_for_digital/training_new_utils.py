#encoding=utf-8
import logging
import os
import os.path
import time 
import numpy as np
import tensorflow as tf
import random
from PIL import Image
 
g_number_def = ['0','1','2','3','4','5','6','7','8','9']
# 图像大小
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
MAX_CAPTCHA = 1
print("Max number of label:", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
 
# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
 
"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""
 
# 文本转向量
char_set = g_number_def 
CHAR_SET_LEN = len(char_set)
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')
 
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map') 
        return k
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector
# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
 
"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""

def list_all_file(rootDir,all_file): 
    if not os.path.exists(rootDir) or not os.path.isdir(rootDir):
        return False
    try:
        for lists in os.listdir(rootDir): 
            path = os.path.join(rootDir, lists) 
            all_file.append(path)
            if os.path.isdir(path): 
                list_all_file(path,all_file) 
    except Exception as e:
        print(e)
        return False
    return True

g_train_all_files = []
list_all_file("./train_set",g_train_all_files)
g_train_index = 0
g_train_set_size = len(g_train_all_files)
g_test_all_files = []
list_all_file("./test_set",g_test_all_files)


 
# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])
    def wrap_gen_captcha_text_and_image(index):
        global g_train_all_files
        global g_train_index
        global g_train_set_size
        file_item = None
        if g_train_index < g_train_set_size:
            file_item = g_train_all_files[g_train_index]
            g_train_index += 1
        else:
            random.seed()
            file_item = random.choice(g_train_all_files)
        base_name = os.path.basename(file_item)
            
        pos = base_name.find(".jpg") 
        file_name = base_name[:pos] 
        name_slice = file_name.split("_")
        text = name_slice[2]

        captcha_image = Image.open(file_item)
        image = np.array(captcha_image)
        if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            return text, image
 
    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image(i)
        image = convert2gray(image)
 
        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = text2vec(text)
 
    return batch_x, batch_y

def get_next_test_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])
 
    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        random.seed()
        file_item = random.choice(g_test_all_files)
        base_name = os.path.basename(file_item)

        pos = base_name.find(".jpg") 
        file_name = base_name[:pos] 
        name_slice = file_name.split("_")
        text = name_slice[2]

        captcha_image = Image.open(file_item)
        image = np.array(captcha_image)
        if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            return text, image
 
    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
 
        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = text2vec(text)
 
    return batch_x, batch_y
 
####################################################################
