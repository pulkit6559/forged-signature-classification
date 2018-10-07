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
forged = []
def create_array():
    for (dirname, dirs, files) in os.walk('.'):
        if dirname[2:] == "forged_bin":
            #fig = plt.figure()
            for filename in files:

                img = Image.open(dirname+"/"+filename)
                img = np.array(img)
                
                f_name.append([int(filename[9:12]), int(filename[7:9]), int(filename[4:7])])
                f_array.append(img.ravel())
                print(img.size)
                forged.append(np.concatenate([f_name, f_array], axis=1))

        """ if dirname[2:] == "genuine_scaled":
            #fig = plt.figure()
            for filename in files:

                img = Image.open(dirname+"/"+filename).convert('L')
                img = np.array(img)
                g_name.append([int(filename[9:12]), int(filename[7:9]), int(filename[4:7])])
                g_array.append([img]) """


if __name__ == "__main__":
    create_array()
    print(forged)
