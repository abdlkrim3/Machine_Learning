import cv2
import numpy as np
from time import sleep

class sign_detection(object):
        def __init__(self, width, height):
                self.scale_percent = 100 # percent of original size
                self.width = int(width * self.scale_percent / 100)
                self.height = int(height *self. scale_percent / 100)
                
                self.dim = (self.width, self.height)
                
                self.sign_detected = False
                
                # load classifier
                self.stop_sign_ENG_cascade = cv2.CascadeClassifier('stopsign_classifier.xml')

                self.IN_image = None
                self.OUT_image = None

                self.is_running = True

                
        def run(self):
                while self.is_running:
                        sleep(0.001)
                        try:
                                if not self.IN_image.any():
                                        continue
                        except:
                                continue
                        
                        #image = cv2.resize(self.IN_image, self.dim)
                        
                        img_filter = cv2.GaussianBlur(self.IN_image, (5, 5), 0)
                        gray_filered = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY)

                        stop_signs_ENG = self.stop_sign_ENG_cascade.detectMultiScale(gray_filered, scaleFactor=1.1, minNeighbors=30, minSize=(60, 60))

                        if len(stop_signs_ENG):
                                print("STOP sign Detected")
                                for (x,y,w,h) in stop_signs_ENG:
                                        x *= int(1 / (self.scale_percent / 100))
                                        y *= int(1 / (self.scale_percent / 100))
                                        w *= int(1 / (self.scale_percent / 100))
                                        h *= int(1 / (self.scale_percent / 100))
                                        self.IN_image = cv2.rectangle(self.IN_image, (x, y), (x+w, y+h), (255, 255, 0), 3)
                                self.sign_detected = True
                        else:
                                self.sign_detected = False

                        self.OUT_image = self.IN_image
                        self.IN_image = None

                
def detect_sign(ORimage):
        scale_percent = 30 # percent of original size
        width = int(ORimage.shape[1] * scale_percent / 100)
        height = int(ORimage.shape[0] * scale_percent / 100)
        
        dim = (width, height)
        image = cv2.resize(ORimage, dim)
        
        sign_detected = False
        # load classifier
        stop_sign_ENG_cascade = cv2.CascadeClassifier('stop_sign_classifier_ENG.xml')

        img_filter = cv2.GaussianBlur(image, (5, 5), 0)
        gray_filered = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY)

        stop_signs_ENG = stop_sign_ENG_cascade.detectMultiScale(gray_filered, scaleFactor=1.05, minNeighbors=20, minSize=(20, 20))

                
        if len(stop_signs_ENG):
                print("STOP sign Detected")
                for (x,y,w,h) in stop_signs_ENG:
                        x *= int(1 / (scale_percent / 100))
                        y *= int(1 / (scale_percent / 100))
                        w *= int(1.2 / (scale_percent / 100))
                        h *= int(1.2 / (scale_percent / 100))
                        ORimage = cv2.rectangle(ORimage, (x, y), (x+w, y+h), (255, 255, 0), 3)
                sign_detected = True
                        
        
        return sign_detected, ORimage
    
