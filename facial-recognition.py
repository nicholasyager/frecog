#!/usr/bin/python2.7

import cv2
import os
import numpy as np
import sys

FACE_CASCADE_PATH = "cascades/haarcascade_frontalface_alt.xml"
EYES_CASCADE_PATH = "cascades/haarcascade_eye_tree_eyeglasses.xml"
TRAINING_PATH = "training.txt"
IDENTITIES_PATH = "identities.txt"
Historic_x = 0
Historic_y = 0
Historic_w = 0
Historic_h = 0

###############################################################################
#
# Functions
#
###############################################################################

def read_csv(training_path):
    """
    Load the training file, and read each image into the program as a matrix.

    Arguments:
        training_path: The path to the training file.
        scale_size: The size to scale the images to.

    """

    trainingFILE = open(training_path, "r")
    indexes = []
    images = []
    for line in trainingFILE:
        image_path = line.strip().split(";")[0]
        subjectid = line.strip().split(";")[1]

        image = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        if (image) is not None:
            image = cv2.resize(image, (150,150))

            cv2.equalizeHist( image, image)
            indexes.append(int(subjectid))
            images.append(image)

    return indexes, images

def read_identities(identity_path):
    identityFile = open(identity_path, "r")
    identities = {}
    for line in identityFile:
        id = line.strip().split(";")[0]
        name = line.strip().split(";")[1]
        identities[id] = name

    return identities

def detect_faces(image):
    """
    Find the location of a face, draw a sqare around it, and send the frame to
    the facial reconition engine.

    Arguments:
        image: The image matrix to detect a face from.
    """
    faces = []
    detected = face_cascade.detectMultiScale(image,scaleFactor= 1.2,
                                    minNeighbors= 2, 
                                    minSize=(100,100))
    try:
        if (detected).all():
            for (x,y,w,h) in detected:
                faces.append((x,y,w,h))
    except AttributeError:
        # No face was detected
        pass
    return faces

def detect_eyes(image):
    """
    Find the location of eyes to properly align the face.

    Arguments:
        image: The image matrix to detect a face from.
    """

    eyes = []
    detected = eyes_cascade.detectMultiScale(image,scaleFactor= 1.2,
                                    minNeighbors= 2, 
                                    minSize=(5,5))


    try:
        if (detected).all():
            for (x,y,w,h) in detected:
                eyes.append((x,y,w,h))
    except AttributeError:
        # No face was detected
        pass
    return eyes

def learnFace(image, id):
    """
    Add the face to the model, the training folder, and the training roster.

    Arguments:
        image: The face image matrix.
        id: The subject id of the person.
    """
    numberOfImage = len([name for name in os.listdir('data/'+str(id)+"/") if os.path.isfile(name)])
    cv2.imwrite("data/"+str(id)+"/"+str(numberOfImage+1)+".bmp", image)

    trainingFile = open(TRAINING_PATH, "a")
    trainingFile.write("data/"+str(id)+"/"+str(numberOfImage+1)+".bmp;"+str(id)+"\n")
    trainingFile.close()

    images.append(image)
    indexes.append(id)
    faceRecog.train(images, np.array(indexes))

### Load facial recognition
indexes, images = read_csv(TRAINING_PATH)
identities = read_identities(IDENTITIES_PATH)
faceRecog = cv2.createEigenFaceRecognizer()


faceRecog.train(images, np.array(indexes))

### Load facial detection

webcamCapture=cv2.VideoCapture()

webcamCapture.open(0)

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eyes_cascade = cv2.CascadeClassifier(EYES_CASCADE_PATH)

faces = []
i = 0

while True:
    retval,retimage=webcamCapture.read() # Load the image
    image=retimage.copy()

    ## Format the raw webcam image.
    image = cv2.flip(image,1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.medianBlur(image, 3)
    cv2.equalizeHist( image)

    if i%1==0:
        faces = detect_faces(image)

    if len(faces) == 0:

    for (x,y,w,h) in faces:

        #y = int((y*0.3)+y)
        #x = int((x*0.1)+x)
        #h = int(h-(h*0.1))
        #w = int(w-(w*0.15))

        threshold = 20

        if x > Historic_x - threshold and x < Historic_x + threshold:
            x = Historic_x
        else:
            Historic_x = x

        if y > Historic_y - threshold and y < Historic_y + threshold:
            y = Historic_y
        else:
            Historic_y = y

        if h > Historic_h - threshold and h < Historic_h + threshold:
            h = Historic_h
        else:
            Historic_h = h

        if w > Historic_w - threshold and w < Historic_w + threshold:
            w = Historic_w
        else:
            Historic_w = w


        ## Faces were detected...
        face = image[y:y+h,x:x+w]
        face = cv2.resize(face, (150,150))
        face = cv2.equalizeHist(face)
       
        ## Find useful tracking features
        

        
        #trackingCorners = cv2.goodFeaturesToTrack(face, 20, 0.01, 10)
        #for corner in trackingCorners:
        #    print(corner)
            #xp = corner[0].split("  ")[1]
            #yp = corner[0].split("  ")[2]

            #cv2.circle(image, (x+xp, y+yp), 3, 255)

        cv2.rectangle(image, (x,y), (x+w,y+h), 255)
        

        #eyes = detect_eyes(face)
        
        #for (xe, ye, we, he) in eyes: 
        #    cv2.rectangle(image, (x+xe,y+ye), (x+xe+we,y+ye+he), 255)

        prediction_label = faceRecog.predict(face)

        #cv2.imshow("Face", face)
        if prediction_label[1] > 3000:
            #cv2.waitKey(1)
        #else:
            #print "Are you "+identities[str(prediction_label[0])]+"?"
            #keyResponse = cv2.waitKey(0)
            #if keyResponse is 121:
            #    learnFace(face, prediction_label[0])
            pass
        try:
            cv2.putText(image, (identities[str(prediction_label[0])] + " - " + 
                    str(prediction_label[1])), 
                    (x,y-10), 
                    cv2.FONT_HERSHEY_PLAIN, 0.8, 255)
            print(identities[str(prediction_label[0])])
        except KeyError:
            cv2.putText(image, "NAME ERROR" , 
                    (x,y-10), 
                    cv2.FONT_HERSHEY_PLAIN, 0.8, 255)
            print("NAME ERROR")
        

    #cv2.imshow("Image_Window", image)
    #cv2.waitKey(1)


    i+=1

