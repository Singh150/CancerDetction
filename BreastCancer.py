import numpy as np#mathemetical calculation
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image#python imaging library
import keras.backend as K
K.set_image_data_format('channels_last')
import os
import keras

modelSavePath = 'D:\\HimaniSingh\\Deployment-flask-master\\Cancer-detection\\my_model3.h5'
numOfTestPoints = 2
batchSize = 16#observation--results
numOfEpoches = 10#iteration--how many code run

classes = []

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
#Crop and rotate image, return 12 images

def getCropImgs(img, needRotations=False):
    z = np.asarray(img, dtype=np.int8)
    c = []
    for i in range(3):
        for j in range(4):
            crop = z[512 * i:512 * (i + 1), 512 * j:512 * (j + 1), :]

            c.append(crop)
            if needRotations:
                c.append(np.rot90(np.rot90(crop)))
    return c
#Get the softmax from folder name
def getAsSoftmax(fname):
    if (fname == 'b'):
        return [1, 0, 0, 0]
    elif (fname == 'is'):
        return [0, 1, 0, 0]
    elif (fname == 'iv'):
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]

# Return all images as numpy array, labels
        # x = np.empty(shape=[19200,512,512,3],dtype=np.int8)
    # y = np.empty(shape=[400],dtype=np.int8)
def get_imgs_frm_folder(path):
    x = []
    y = []
    cnt = 0
    for foldname in os.listdir(path):
        for filename in os.listdir(os.path.join(path, foldname)):
            img = Image.open(os.path.join(os.path.join(path, foldname), filename))
            # img.show()
            crpImgs = getCropImgs(img)
            cnt += 1
            if cnt % 10 == 0:
                print(str(cnt) + " Images loaded")
            for im in crpImgs:
                x.append(np.divide(np.asarray(im, np.float16), 255.))
                y.append(getAsSoftmax(foldname))
    print("Images cropped")
    print("Loading as array")
    return x, y, cnt

#predicting the images
def predict(img, savedModelPath, showImg=True):
    model = load_model(savedModelPath)
    # if showImg:
    # Image.fromarray(np.array(img, np.float16), 'RGB').show()

    x = img
    if showImg:
        Image.fromarray(np.array(img, np.float16), 'RGB').show()
    x = np.expand_dims(x, axis=0)

    softMaxPred = model.predict(x)
    print("prediction from Algo: " + str(softMaxPred) + "\n")
    probs = softmaxToProbs(softMaxPred)

    # plot_model(model, to_file='Model.png')
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    maxprob = 0
    maxI = 0
    for j in range(len(probs)):
        # print(str(j) + " : " + str(round(probs[j], 4)))
        if probs[j] > maxprob:
            maxprob = probs[j]
            maxI = j
    # print(softMaxPred)
    print("prediction index: " + str(maxI))
    return maxI, probs

#Softmax is often used in neural networks, to map the non-normalized output of a network to a probability distribution
def softmaxToProbs(soft):
    z_exp = [np.math.exp(i) for i in soft[0]]
    sum_z_exp = sum(z_exp)
    return [(i / sum_z_exp) * 100 for i in z_exp]

#predicting images
def predictImage(img_path, arrayImg=None, printData=True):
    crops = []
    if arrayImg == None:
        img = keras.utils.load_img(img_path)
        crops = np.array(getCropImgs(img, needRotations=False), np.float16)
        crops = np.divide(crops, 255.)
    Image.fromarray(np.array(crops[0]), "RGB").show()

    classes = []
    classes.append("Benign")
    classes.append("InSitu")
    classes.append("Invasive")
    classes.append("Normal")

    compProbs = []
    compProbs.append(0)
    compProbs.append(0)
    compProbs.append(0)
    compProbs.append(0)

    for i in range(len(crops)):
        if printData:
            print("\n\nCrop " + str(i + 1) + " prediction:\n")

        ___, probs = predict(crops[i], modelSavePath, showImg=False)

        for j in range(len(classes)):
            if printData:
                print(str(classes[j]) + " : " + str(round(probs[j], 4)) + "%")
            compProbs[j] += probs[j]

    if printData:
        print("\n\nAverage from all crops\n")

    for j in range(len(classes)):
        if printData:
            print(str(classes[j]) + " : " + str(round(compProbs[j] / 12, 4)) + "%")

import tkinter.filedialog as f
import tkinter 
win=tkinter.Tk()
predictImage(f.askopenfilename(filetypes=()))
win.destroy()
