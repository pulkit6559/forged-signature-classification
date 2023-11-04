from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import array
import operator
path = './'

f_name = []
f_array = []
g_name = []
g_array = []
baseheight = 200
basewidth = 400


def homogenize(img):
    height = img.size[1]
    width = img.size[0]
    img = np.array(img)
    print(width , height)

    if width/height<2:
        n_width = 2*height
        padding = int((n_width - width)/2)
        pix = np.sum(img[:, 0])/(height)

        pad = np.ones([height, padding])*pix

        return np.concatenate((pad, img, pad), axis=1)
    elif width/height>2:
        n_height = int(width/2)
        padding = int((n_height - height)/2)
        pix = np.sum(img[0, :])/(width)

        pad = np.ones([padding, width])*pix

        return np.concatenate((pad, img, pad), axis=0)
    else :
        return img


def crop(img):
    height = img.size[1]
    width = img.size[0]
    up = 0
    down = height
    left = 0
    right = width
    img = np.array(img)
    
    for i in range(0,int(height/2)):
        if np.sum(img[i,:])==0:
            up = up + 1

        if np.sum(img[height-i-1,:])==0:
            down = down-1
    for i in range(0, int(width/2)):
        if np.sum(img[:, i]) == 0:
            left = left + 1

        if np.sum(img[:, width-i-1]) == 0:
            right = right-1
    return img[up:down+1,left:right+1]
        
"""
def scale(img):
    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize, baseheight), Image.ANTIALIAS)
    return img
"""
def create_dir():

    for (dirname, dirs, files) in os.walk('.'):
        if dirname[2:] == "forged":
            #fig = plt.figure()
            for filename in files:
                print(filename)
                #img = Image.open(dirname+"/"+filename).convert('1')
                img = cv2.imread(dirname+"/"+filename,0)
                
                print(type(img))
                img = Image.fromarray(img)
                img = crop(img)
                ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                
                #f_name.append([filename[4:7], filename[9:12], filename[7:9]])
                #f_array.append(img)
                print(dirname+"/"+filename)
                img = Image.fromarray(img)
                img = homogenize(img)
                img = Image.fromarray(img)
                img = img.convert("1")
                img = img.resize((basewidth, baseheight), Image.ANTIALIAS)
                img.save("./"+"forged_bin"+"/"+filename, "PNG")
            
        if dirname[2:] == "genuine":
        #fig = plt.figure()
            for filename in files:
                print(filename)
                #img = Image.open(dirname+"/"+filename).convert('1')
                img = cv2.imread(dirname+"/"+filename, 0)

                print(type(img))
                img = Image.fromarray(img)
                img = crop(img)
                ret, img = cv2.threshold(
                    img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                #f_name.append([filename[4:7], filename[9:12], filename[7:9]])
                #f_array.append(img)
                print(dirname+"/"+filename)
                img = Image.fromarray(img)
                img = homogenize(img)
                img = Image.fromarray(img)
                img = img.convert("1")
                img = img.resize((basewidth, baseheight), Image.ANTIALIAS)
                img.save("./"+"genuine_bin"+"/"+filename, "PNG")

            

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))

#NFI-00101001.png

def create_array():
    for (dirname, dirs, files) in os.walk('.'):
        if dirname[2:] == "forged_scaled":
            #fig = plt.figure()
            for filename in files:

                img = Image.open(dirname+"/"+filename).convert('L')
                img = np.array(img)
                f_name.append([int(filename[9:12]),int(filename[7:9]) , int(filename[4:7])])
                f_array.append([img])
                #forged = np.concatenate([f_name, f_array], axis=1)
        
        if dirname[2:] == "genuine_scaled":
            #fig = plt.figure()
            for filename in files:

                img = Image.open(dirname+"/"+filename).convert('L')
                img = np.array(img)
                g_name.append([int(filename[9:12]),int(filename[7:9]) , int(filename[4:7])])
                g_array.append([img])
                #genuine.append(np.concatenate([g_name, g_array], axis = 1))

if __name__ == '__main__':
    create_dir()
    #f_name.sort(key=operator.itemgetter(0, 1))
    #print(genuine)