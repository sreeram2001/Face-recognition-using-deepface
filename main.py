# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:28:37 2021

@author: SREERAM S
"""

""" Face Recognition System Using Deepface """
import matplotlib.pyplot as plt
import cv2 as cv
from deepface import DeepFace as df

img1 = '1555f4d5-3831-4315-8044-4f2a1e8d83b0.jpg'
img2 = '2e64ad4d-8109-4936-b77a-7016cd520ede.jpg'

image_1 = cv.imread(img1)
plt.subplot(2,1,1)
plt.imshow(image_1[:,:,::-1])
plt.show()

image_2 = cv.imread(img2)
plt.subplot(2,1,2)
plt.imshow(image_2[:,:,::-1])        #[:,:,::-1] to get the original image itself maintains saturation
plt.show()


""" library model named VGG Face to analyse our face"""
#imports pre-trained deep learning networks, pre-trained weights

result = df.verify(image_1,image_2)
print("Face Matching :",result['verified'])


#To display the Characterisitcs of the Face
obj = df.analyze(img_path='2e64ad4d-8109-4936-b77a-7016cd520ede.jpg',actions=['age','gender','race','emotion'])
print("Age: ",obj['age'])
print("Gender: ",obj['gender'])
print("Race: ",obj['race'])
print("Emotion: ",obj['emotion'])
