import os
import random
import shutil

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

def getData(src_path):

    dest_dir = src_path+'/query'
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    img_list = get_imlist(src_path+'/bounding_box_test')
    # img_list = get_imlist(src_path)
    random.shuffle(img_list)
    le = int(len(img_list) * 0.7)
    for f in img_list[le:]:
        shutil.move(f, dest_dir)



def SplitImg(filePath):
    getData(filePath)


def MoveAn(filePathAn,filePathImg):
    Imgs=os.listdir(filePathImg)
    if not os.path.isdir(filePathAn+'val'):
        os.mkdir(filePathAn+'val')

    for file in os.listdir(filePathAn):
        #print(filePathAn,filePathImg)
        #print(os.path.join(filePathAn,file),os.path.join(filePathAn+'val',file))
        if file[:-4]+'.jpg' in Imgs:

            shutil.move(os.path.join(filePathAn,file),os.path.join(filePathAn+'val',file))

if __name__=='__main__':
    filePath=r''# 换成你的数据集

    #拆分的数据集
    SplitImg(filePath)

