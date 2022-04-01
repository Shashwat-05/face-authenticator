import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# enter the name similar to the user's data collected
NAME = input("enter your name : ")

#the path of user's images directory
data_path = f'./{NAME}/'

#will store individual image name into the list
onlyfiles = [fi for fi in listdir(data_path) if isfile(join(data_path, fi))]

# Create empty arrays for training data and labels
Training_Data, Labels = [], []

#filling the above arrays with training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i] #individual image path
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #read the image
    Training_Data.append(np.asarray(images, dtype=np.uint8)) #convert img to int8 and store into arr
    Labels.append(i)


# convert labels into int32
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer model (empty)
model=cv2.face.LBPHFaceRecognizer_create()


# training the model
model.train(np.asarray(Training_Data), np.asarray(Labels)) 
print("Model trained sucessefully ....")

#saving the trained model
model.save(f'{NAME}ttt.yml')