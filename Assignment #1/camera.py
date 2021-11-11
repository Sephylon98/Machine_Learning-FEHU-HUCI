# Written by: Mohamed Ashraf Mohamed Shebl , Student ID: 19037

# used packages:
import time as t
import cv2
from matplotlib import pyplot as plt
from matplotlib import image
from numpy import asarray
import numpy as np
from PIL import Image


# program interface:
print("Welcome user."); t.sleep(0.1)
print("Please choose of the following options:"); t.sleep(0.5)
print ("Enter '1' to take a picture from your camera, Enter '2' to read from camera from a specific location on your system.") 
user_input = int(input('Entery option: ')); t.sleep(0.5)

if user_input == 1:
    # print("Pillow Version: ",PIL.__version__)
    counter  = 0 #counter for showing the message on frame
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #take image from camera
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL) #name of the window containing the video frames
    cv2.resizeWindow('frame', 800,800)  #name , width , height
    while(cap.isOpened()):
        ret, frame = cap.read() #ret returns state value from getting camera frame either true or false
        if counter<= 40:
            cv2.putText(frame,'Press Q to exit camera',(5,45),1,2,(255,0,0),2,2)
        # print(frame.shape)
        counter += 1
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    #Print state:
    #1. Transform picture to a numpy array and store it in variable data
    #2. draw the pixels value stored inside data numpy array
    #3. show the image and you can save it on your disk or not because we will save it anyway

    data = asarray(frame)
    print(data) 
    #save numpy array into a file:
    np.save('np_camera',data) 
    # display the array of pixels as an image
    plt.imshow(data)
    plt.show()
    t.sleep(5)
    camera = Image.fromarray(data).save('camera.png')
    #convert color image to grey scale image
    image = np.array(Image.fromarray(data).convert('L'))
    gray_image= Image.fromarray(image).save('grey_camera.png')
        

elif user_input == 2:
    image = image.imread('koala.jpeg')
    print(image)
    #save numpy array into a file:
    np.save('np_koala',image)
    #display the array of pixels as an image
    plt.imshow(image)
    plt.show()
    t.sleep(5)
    # convert color image to grey scale image
    image = np.array(Image.open('koala.jpeg').convert('L'))
    gray_image= Image.fromarray(image).save('gray_koala.png')

else:
    print("Please enter a correct entery next time! ")
    t.sleep(0.5)        



 

