import os

#更改数据集中的图片名字，格式为x_c1_y.jpg
def batchrename_croppeddataset(path):
    os.chdir(path)
    print("   CHANGE DIRECTORY TO THE PATH   ")
    filelist = os.listdir(path)
    print("   File List   ")
    for filenames in filelist:
        if filenames =="list.txt":
            continue
        os.chdir(path+"/"+filenames)
        files = os.listdir(path+"/"+filenames)
        print(" 开始修改"+filenames+"内的图片")
        for fi in files:
            newname = filenames+"_c1_" + fi
            os.rename(fi, newname)

#
def batchrename_quary(path):
    os.chdir(path)
    print("   CHANGE DIRECTORY TO THE PATH   ")
    filelist = os.listdir(path)
    print("   File List   ")
    cnt = 1 #计数用
    crm = 1 #摄像头编号
    idnm = ''
    for filenames in filelist:
        if filenames =="list.txt":
            continue
        pos = filenames.find("_")
        id1,_,id2=filenames.split("_",pos)

        print(" 开始修改"+filenames+"内的图片")
        if cnt%5==1:
            cnt = 1
            crm = 1
            idnm=id1

        # if cnt%10==1:
        #     crm += 1

        newname = id1+"_c"+str(crm)+"_" + id2
        os.rename(filenames, newname)
        cnt += 1
        crm += 1

        # if cnt%3==1:
        #     newname = id1+"_c1_" + id2
        #     os.rename(filenames, newname)
        # elif (cnt%3==2) :
        #     newname = id1 + "_c2_" + id2
        #     os.rename(filenames, newname)
        # else:
        #     newname = id1 + "_c3_" + id2
        #     os.rename(filenames, newname)
        # cnt +=1

#更改数据集中的图片名字，格式为x_c1_y.jpg
# batchrename_croppeddataset(r"E:\Desktop\Vessel_photo\cropped\warship")

#更改quary文件夹的摄像头
batchrename_quary("/home/server07/桌面/cui_proj/dataset/dukemtmcreid/bounding_box_train")