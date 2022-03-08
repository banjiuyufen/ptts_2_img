import os
import os.path as osp
import numpy as np
import numpy.linalg as npl
import cv2
from PIL import Image
import struct
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import ctypes

def read_ptts_from_dir(ptts_dir):
    with open(ptts_dir, 'rb') as f:
        header_size = struct.unpack('l', f.read(4))[0]
        #print(header_size)
        Format_code = f.read(8)
        #print(Format_code)
        Ill = f.read(header_size - 54)
        #print(Ill)
        code_type = f.read(20)
        #print(code_type)
        code_length = struct.unpack('h',f.read(2))[0]
        #print(code_length)
        data_type = f.read(20)
        #print(data_type)
        sample_length = struct.unpack('i', f.read(4))[0]
        #print(sample_length)
        page_index = struct.unpack('i',f.read(4))[0]
        #print(page_index)
        stroke_num = struct.unpack('i', f.read(4))[0]
        #print('笔画数：%d' %stroke_num)
        traj = []
        point_num = []
        for i in range(stroke_num):
            pointnum = struct.unpack('h', f.read(2))[0]
            point_num.append(pointnum)
            #print('采样点数:%d' %pointnum)
            #traj = []
            for j in range(pointnum):
                x = struct.unpack('H', f.read(2))[0]
                y = struct.unpack('H', f.read(2))[0]
                traj.append([x, y])
        #print(traj)
        line_num = struct.unpack('H', f.read(2))[0]
        #print('行数:%d' %line_num)
        tagcode = []
        char_stroke_index = []
        line_char_nmu_index = []
        char_stroke_num_index = []
        for i in range(line_num):
            line_stroke_nmu = struct.unpack('H', f.read(2))[0]
            #print('行笔画数：%d' %line_stroke_nmu)
            line_stroke_index = []
            for j in range(line_stroke_nmu):
                stroke_index = struct.unpack('H', f.read(2))[0]
                line_stroke_index.append(stroke_index)
            #print(line_stroke_index)
            line_char_nmu = struct.unpack('H', f.read(2))[0]
            line_char_nmu_index.append(line_char_nmu)
            #print('该行有：%d字符' %line_char_nmu)
            for char in range(line_char_nmu):
                tag_code = f.read(code_length)
                tag_code = tag_code.decode('gb18030')
                tag_code = tag_code.replace('\x00', '')
                tagcode.append(tag_code)
                char_stroke_num = struct.unpack('H', f.read(2))[0]
                char_stroke_num_index.append(char_stroke_num)
                #print('字符笔画数为：%d' %char_stroke_num)
                char_stroke_index_1 = []
                for index in range(char_stroke_num):
                    char_stroke_index_0 = struct.unpack('H', f.read(2))[0]
                    char_stroke_index_1.append(char_stroke_index_0)
                char_stroke_index.append(char_stroke_index_1)

        f.close()
    return stroke_num, point_num, traj, line_num, line_char_nmu_index, char_stroke_num_index, char_stroke_index, tagcode



def drawPtts(page_info, path):
    stroke_num = page_info[0]
    point_num = page_info[1]
    traj = page_info[2]
    line_num = page_info[3]
    line_char_nmu = page_info[4]
    char_stroke_num = page_info[5]
    char_stroke_index = page_info[-2]
    tagcode = page_info[-1]
    #print(traj)
    aaa = 0
    #打印出文本行内容
    #print("文本内容为：")
    for word in tagcode:
        print(word, end="")
    print('\n')
    count = 0
    star = 0
    begin = 0
    for line_index in range(line_num):
        char_num = line_char_nmu[line_index]
        #print('现在是第%d行,该行有%d字符。' %(line_index+1, char_num))
    # 提取每行字符的笔画数
        char_stroke_list = char_stroke_num[count:char_num + count]
        count = char_num + count

        #print(char_stroke_list)
        #提取每个字符的笔画数 列表长度代表该字符的笔画数
        for char_index in char_stroke_list:
            points_list = point_num[star:star+char_index]
            star = star + char_index
            #print(points_list)
            #提取每个笔画的采样点

            for point in points_list:
                coordinates_list = traj[begin:begin + point]
                begin = begin + point
                #print(coordinates_list)
                #print(len(coordinates_list))
                x_list = []
                y_list = []
                for xy in coordinates_list:
                    x_list.append(xy[0]/10)
                    y_list.append(xy[1]/10)
                    img = plt.plot(x_list, y_list, 'black', linewidth=2.0)
                    #plt.scatter(x_list, y_list, color='b')
                aaa+=1
            ax = plt.gca()
            ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部
            ax.yaxis.set_ticks_position('left')  # 将y轴的位置设置在右边
            ax.invert_yaxis()  # y轴反向
            plt.axis('off')
            plt.savefig(path + "%d.png"%aaa)
            plt.show()


root = osp.join('D:/appa')
train_dir = osp.join(root, 'cnntrain')
#test_dir = osp.join(root, 'cnntest')
train_dataset = os.listdir(train_dir)
#test_dataset = os.listdir(test_dir)
train_parse_dir = osp.join(root, 'trn/')
#test_parse_dir = osp.join(root, 'tst/')

for file_name in os.listdir(train_dir):
    if file_name.endswith('.ptts'):
        file_path = os.path.join(train_dir, file_name)
        drawPtts(read_ptts_from_dir(file_path), train_parse_dir)

