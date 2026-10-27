import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import re
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def get_images(trainloader,writer,net):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels, _, _= next(dataiter)
    # images, labels = data_all.image,data_all.label

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)
    writer.add_graph(net, images)
    writer.add_graph(net,input_to_model = torch.rand(1,3,256, 128))


    # writer.close()

def imshow(path, title=None,width=512,height=512):
    """Imshow for Tensor."""
    im = Image.open(path).resize((width,height))
    # im = plt.imread(im)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated


def show_pic_rank(img_path,img_list,img_label,line=1):
    gallary_path= img_path+'/'+'bounding_box_test'
    query_path = img_path + '/' + 'query'

    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 11, 1)
    ax.axis('off')
    imshow(query_path+'/'+img_list[0], 'query')
    for i in range(10):
        ax = plt.subplot(1, 11, i + 2)
        ax.axis('off')
        img_path = gallary_path+'/'+img_list[i+1]
        imshow(img_path)
        if img_label[i] == 1:
            ax.set_title('%d' % (i + 1), color='green')
        else:
            ax.set_title('%d' % (i + 1), color='red')
        print(img_path)
    fig.savefig("show1.png")


def show_pic_rank_2(img_path,img_list,img_label):

    gallary_path = img_path + '/' + 'bounding_box_test'
    query_path = img_path + '/' + 'query'
    for ind in range(20):
        fig = plt.figure(figsize=(32, 16),dpi=100)
        for line in range(4):
            ax = plt.subplot(4, 11, 11 * (line) + 1)
            ax.axis('off')
            imshow(query_path + '/' + img_list[line+4*(ind)][0], 'query')
            ax.text(0.5, -0.1, str(img_list[line+4*(ind)][0]), size=6, ha="center",
                    transform=ax.transAxes, color='blue')
            for i in range(10):
                ax = plt.subplot(4, 11, 11 * (line) + i + 2)
                ax.axis('off')
                img_path = gallary_path + '/' + img_list[line+4*(ind)][i + 1]
                imshow(img_path)
                if img_label[line+4*(ind)][i] == 1:
                    ax.set_title('%d' % ( i + 1), color='green')
                    ax.text(0.5, -0.1, str(img_list[line+4*(ind)][i + 1]), size=6, ha="center",
                            transform=ax.transAxes, color='green')
                else:
                    ax.set_title('%d' % ( i + 1), color='red')
                    ax.text(0.5, -0.1, str(img_list[line+4*(ind)][i + 1]), size=6, ha="center",
                            transform=ax.transAxes, color='red')
        fig.savefig("")



