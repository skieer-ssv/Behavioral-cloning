import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.utils import shuffle
import random
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam
def getName(filePath):
    return filePath.split('\\')[-1]


def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    #### REMOVE FILE PATH AND GET ONLY FILE NAME
    # print(getName(data['center'][0]))
    data['Center'] = data['Center'].apply(getName)
    # print(data.head())
    print('Total Images Imported', data.shape[0])
    return data


def balanceData(data,display=True):
    nBin = 31
    samplesPerBin = 1000
    hist, bins = np.histogram(data['Steering'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], (nBin))
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()

def loadData(path, data):
  imagesPath = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    imagesPath.append(f'{path}/{indexed_data[0]}')
    steering.append(float(indexed_data[3]))
  imagesPath = np.asarray(imagesPath)
  steering = np.asarray(steering)
  return imagesPath, steering

def augmentImage(imgPath,steering):
    img =  mpimg.imread(imgPath)
    #pan
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    #zoom
    # if np.random.rand() < 0.5:
    #     zoom = iaa.Affine(scale=(1, 1.2))
    #     img = zoom.augment_image(img)
    #brightness
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    #flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering

def preProcess(img):
    img = crop(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


def laneDetection(img):
    hls_img = img

    # Compute a binary thresholded image where yellow is isolated from HLS components
    img_hls_yellow_bin = np.zeros_like(hls_img[:, :, 0])
    img_hls_yellow_bin[((hls_img[:, :, 0] >= 15) & (hls_img[:, :, 0] <= 35))
                       & ((hls_img[:, :, 1] >= 30) & (hls_img[:, :, 1] <= 204))
                       & ((hls_img[:, :, 2] >= 115) & (hls_img[:, :, 2] <= 255))
                       ] = 1

    # Compute a binary thresholded image where white is isolated from HLS components
    img_hls_white_bin = np.zeros_like(hls_img[:, :, 0])
    img_hls_white_bin[((hls_img[:, :, 0] >= 0) & (hls_img[:, :, 0] <= 255))
                      & ((hls_img[:, :, 1] >= 200) & (hls_img[:, :, 1] <= 255))
                      & ((hls_img[:, :, 2] >= 0) & (hls_img[:, :, 2] <= 255))
                      ] = 1

    # Now combine both
    img_hls_white_yellow_bin = np.zeros_like(hls_img[:, :, 0])
    img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1

    return img_hls_white_yellow_bin



def crop(image, top_percent=0.35, bottom_percent=0.1):
    """
    Crops an image according to the given parameters
    Args:
     - image:
        source image
     - top_percent:
        The percentage of the original image will be cropped from the top of the image
     - bottom_percent:
        The percentage of the original image will be cropped from the bottom of the image
     Return:
      - The cropped image
    """
    assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]

def random_shear(image, steering_angle, shear_range=200):
    """
    Args:
     - image:
        Source image on which the shear operation will be applied
     - steering_angle:
        The steering angle of the image
     - shear_range:
        Random shear between [-shear_range, shear_range + 1] will be applied
    Return:
        The image generated by applying random shear on the source image
    """
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle


def random_rotation(image, steering_angle, rotation_amount=15):
    """
    Args:
        - image:
        - steering_angle:
        - rotation_amount:
    Return:
        - Randomly rotated image and steering_angle
    """
    angle = np.random.uniform(-rotation_amount, rotation_amount + 1)
    rad = (np.pi / 180.0) * angle
    return rotate(image, angle, reshape=False), steering_angle + (-1) * rad





def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
            yield (np.asarray(imgBatch), np.asarray(steeringBatch))


def createModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')
    return model
