from sign_detection_module import sign_detection
import lane_detection_module as ld
import numpy as np
import threading
import Control
import cv2
import sys
import time


class lane_detection(object):
    def __init__(self):
        """ Variables """
        self.latestImage = None
        self.cap = cv2.VideoCapture("http://192.168.137.19:8080/video")
        #self.cap = cv2.VideoCapture(0)
        
        cv2.namedWindow('Lane detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Lane detection', 640, 480)

        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(3, 640)  # width=1920
        self.cap.set(4, 480)  # height=1080
        
        self.kernel_size = 19
        self.low_threshold = 0
        self.high_threshold = 20

        self.fps = 0

        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        print((self.height, self.width))

        self.Sd = sign_detection(self.width, self.height)
  
    def run(self):
        start_height = 280 # Scan index row 235

        # Run sign detection thread
        x = threading.Thread(target=self.Sd.run)
        x.start()
        
        Auto = False
        while cv2.getWindowProperty('Lane detection', 0) >= 0:
            # Start time
            start = time.time()

            # Only run loop if we have an image
            OK, self.latestImage = self.cap.read()
            
            if OK:
                # Rotate frame
                self.latestImage = cv2.rotate(self.latestImage, cv2.ROTATE_90_CLOCKWISE)

                # step 1: detect Sign
                self.Sd.IN_image = self.latestImage
                if self.Sd.sign_detected:
                    self.latestImage = self.Sd.OUT_image
                    Control.Stop()
                    Auto = False
                
                # step 2: detect binary lane markings

                self.blurImage = ld.gaussian_blur(self.latestImage, self.kernel_size)
                
                light_black = (100, 100, 100)
                dark_black = (0, 0, 0)

                self.blurImage = cv2.inRange(self.blurImage, dark_black, light_black)
                #self.edgeImage = ld.canny(self.blurImage, self.low_threshold, self.high_threshold)
                self.blurImage = 255 - self.blurImage 
                
                # step 3: Define region of interest for cropping
                height = self.latestImage.shape[0]
                width = self.latestImage.shape[1]

                
                vertices = np.array( [[
                        [5*width/6, 2*height/6],
                        [width/6, 2*height/6],
                        [20, height-10],
                        [width-20, height-10]
                ]], dtype=np.int32 )
            
                self.maskedImage = ld.region_of_interest(self.blurImage, vertices)

                
                ret, thresh = cv2.threshold(self.maskedImage, 220, 255, cv2.THRESH_BINARY_INV)
                
                signed_thresh = thresh[start_height].astype(np.int16)  # select only one row
                diff = np.diff(signed_thresh)  # The derivative of the start_height line

                points = np.where(np.logical_or(diff > 200, diff < -200)) #maximums and minimums of derivative
                
                if len(points) > 0 and len(points[0]) > 1: # if finds something like a black line
                    middle = int(np.mean(points[0]))
                    
                    #print("Mean point : " + str(middle))

                    cv2.circle(self.latestImage, (points[0][0], start_height), 5, (255,0,0), -1)
                    cv2.circle(self.latestImage, (points[0][-1], start_height), 5, (255,0,0), -1)
                    cv2.circle(self.latestImage, (int(middle), int(start_height)), 10, (0,0,255), -1)

                    cv2.putText(self.latestImage, text= "FPS : " + str(self.fps), org=(10,20),
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1/2, color=(0,0,0),
                    thickness=1, lineType=cv2.LINE_AA)

                    if Auto:
                        thr = 80
                        Control.Move(middle, width, thr)
                    else:  
                        Control.Stop()
                        pass

                # End time
                end = time.time()
                if end != start:
                    self.fps = int(1/(end-start))
                    
                cv2.imshow('Lane detection', self.latestImage)
   

            # press 'Q' if you want to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting......")
                print("Good By !!")
                self.cap.release()
                break
            elif cv2.waitKey(1) & 0xFF == ord('a'):
                print("Mode changed")
                Auto = not Auto

if __name__ == '__main__':
    Ld = lane_detection()
    Ld.run()
    cv2.destroyAllWindows()
