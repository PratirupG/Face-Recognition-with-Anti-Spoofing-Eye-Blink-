from cv2 import cv2 #Import the computer vision lobrary Open cv
import numpy as np 
import pandas as pa 
import os 
import face_recognition #Import face recognization library (trained using deep learning )
import warnings
warnings.filterwarnings('ignore') #Avoid warnings in jupyter notebook.
import pickle

class TRAINING_:
    """
            Class TRAINING functonality(
                ->Training all images
                ->Add new image encoding 
                ->Prediction (Face Recognize)
                ->Load previously trained face encodings
            )
    
    """

    def __init__(self,train=True):
        self.database = r'C:\Users\DHIRAJ\Face Recog New\Database'
        
        if train:
            self.known_faces = []#Store all the face encoding(information) of the training data 
            self.known_face_names = []#Store all the names of the person of training data
            self.__encode_train_face(self.database)
        else:
            self.known_faces,self.known_face_names = self.load_encodings()
    
    def new_image(self,new_image,name):
        """
                This function will find the encoding of new image and will  append it on
                previous train encodings.
        """
        try:
            self.known_faces,self.known_face_names = self.load_encodings()
            
            path = r'C:\Users\DHIRAJ\Face Recog New'#New image save location   
            #Find face enconding
            face_new = face_recognition.load_image_file(os.path.join(path,new_image))
            face_encoding_new = face_recognition.face_encodings(face_new)[0] 
            face_new_name = name
            self.known_faces.append(face_encoding_new)
            self.known_face_names.append(face_new_name)
        except:
            print('Error in Function new_image')    

        ##Save the encondings override the existing file 
        #Store the list in file after training 
        with open("face_encodings_train.txt", "wb") as fp:
            pickle.dump(self.known_faces, fp)
        with open("face_train_names.txt", "wb") as fp:
            pickle.dump(self.known_face_names, fp)
        

    def __encode_train_face(self,path):
        """
                This function will be called when Train=True  this function is for training the whole 
                data.
        """
        try:
            print('TRAINING STARTED-------------------------------------------------------')
            all_images = os.listdir(path) #Store all time images of training database 
        
            #print(all_images)
            for image in all_images:
                #Iterate over all time images of training data using for loop
                face_image_name = image.split(sep='.')[0] #Get the name of person from datavse
                self.known_face_names.append(face_image_name) #Append the facename to know_face_names list
                load_image = face_recognition.load_image_file(os.path.join(path,image))  #Load the image
                #load_image = cv2.resize(load_image,(256,256)) #Resize the image
                encode_image = face_recognition.face_encodings(load_image)[0]#Find the face encoding(information) using face_recognization library
                self.known_faces.append(encode_image) #Append the face encoding to known_faces list 
        except:
            print('Error during training')      

        #Store the list in file after training 
        with open("face_encodings_train.txt", "wb") as fp:
            pickle.dump(self.known_faces, fp)
        with open("face_train_names.txt", "wb") as fp:
            pickle.dump(self.known_face_names, fp)
                
    def load_encodings(self):
        with open("face_encodings_train.txt", "rb") as fp:
            known_faces = pickle.load(fp)
        with open("face_train_names.txt", "rb") as fp:
            known_face_names = pickle.load(fp)
        
        return known_faces,known_face_names
    
    def face_recognize(self,image): 
        
        try:
            frame_s = image
            #frame_s = cv2.resize(image,(256,256))
            frame_rgb = cv2.cvtColor(frame_s,cv2.COLOR_BGR2RGB)
            
            ## Find econding of each faces in the frame 
            face_locations = face_recognition.face_locations(frame_rgb)
            face_encodings = face_recognition.face_encodings(frame_rgb,face_locations)
            face_names = []
            
            print('Face Encodeings ',face_encodings[0].shape)
            print('Know Face',self.known_faces[0].shape)

            for face_encode,face_location in zip(face_encodings,face_locations):
                print('Sucess')
                #Compare faces and find the similarity of two faces
                matches = face_recognition.compare_faces(self.known_faces,face_encode)
                name='Unkown'
                face_similarity = face_recognition.face_distance(self.known_faces,face_encode)
                best_face = np.argmin(face_similarity)
                
                if matches[best_face]:
                    name = self.known_face_names[best_face]
                
                face_names.append(name)
        except:
            print('Error in function face_recognize')

        return face_names #Return the name of the face 


