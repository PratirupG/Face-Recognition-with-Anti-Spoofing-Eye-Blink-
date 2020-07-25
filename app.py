from cv2 import cv2
import numpy as np 
import pandas as pa 
import os 
import face_recognition 
import warnings
warnings.filterwarnings('ignore')
import pickle
from face_api import TRAINING_
from imutils.video import FileVideoStream
from scipy import ndimage
from face_spoof import SPOOF_DETECTION

#Load the model
obj = TRAINING_(train=False) # Add new data ijn database and set train=True for train again
known_faces,known_face_names = obj.load_encodings()

class FACE_VERIFICATION:

    def add_new_image(self,img,name):
        """
            Add new image to the database for training the face_recognition model

        :param img:Image of the person
        :param name :Name of the Person
        """
        new = img
        obj.new_image(new,name)

    def face_verification(self,path):
        """
            Face verification from input file video Stream

            :param path: Location of the video
            :return  name: Recognized person's name
        """

        video_path = path
        result = True
        while (result):
            vs = FileVideoStream(video_path).start()
            frame = vs.read()

            frame = ndimage.rotate(frame, 180)
            frame = ndimage.rotate(frame, 90)
            cap_image = frame
            check = SPOOF_DETECTION().spoof_detection(video_path)
            result = False
            vs.stop()
        cv2.destroyAllWindows()

        if check == True:
            name = obj.face_recognize(cap_image)
            print('Recognized image {}'.format(name))
        else:
            print('SPOOFING ')
