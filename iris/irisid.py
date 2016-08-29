import numpy as np
import cv2, os
import time
from PIL import Image
import PIL.ImageOps
from matplotlib import pyplot as plt

#def nothing(x):
 #   pass

class Iris:
    _path = ''


    def __init__(self, path=_path):
        self.path = path
        self.load_image(self.path)

    def preprocess(self):

        for image in self.images:

            blur = cv2.GaussianBlur(image, (5, 5), 0)
            #blur = cv2.bilateralFilter(image, 9, 75, 75)
            ret3, th3 = cv2.threshold(blur, 26, 255, cv2.THRESH_BINARY_INV)
            thres_image = [th3]
            #thres_image_1 = 0.5 * thres_image

            #edges = []
            edges = cv2.Canny(image, ret3, ret3/2 )

            contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #contour_max = contours[max(len(contours))]

            for i in contours:
                print len(i)



            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1000,
                                       param1=ret3, param2=15, minRadius=0, maxRadius=0)

            if circles is None:
                print "No circles found"
            else:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # draw the outer circle
                    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

                cv2.imshow('detected circles', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Generate trackbar Window Name
            #cv2.namedWindow('WindowName')

            # Make Window and Trackbar
            #cv2.createTrackbar('TrackbarName', 'WindowName', 0, 255, nothing)

            # Allocate destination image
            #Threshold = np.zeros(image.shape, np.uint8)

            # Loop for get trackbar pos and process it
            #while True:
                # Get position in trackbar
            #    TrackbarPos = cv2.getTrackbarPos('TrackbarName', 'WindowName')
                # Apply threshold

             #   blur = cv2.GaussianBlur(image, (7, 7), 0)
                # Otsu nao funciona...
#                ret3, th3 = cv2.threshold(blur, TrackbarPos, 255, cv2.THRESH_BINARY)

                #cv2.threshold(image, TrackbarPos, 255, cv2.THRESH_BINARY, Threshold)
                # Show in window
 #               cv2.imshow('WindowName', th3)

                # If you press "ESC", it will return value
  #              ch = cv2.waitKey(5)
   #             if ch == 27:
    #                break

     #       cv2.destroyAllWindows()

            #edge_max = edges[max(len(edges))]

            #cv2.imshow('image', edges)
            #ch = cv2.waitKey(50)
            #if ch == 27:
            #    break
            #time.sleep(10)
            #for i in xrange(1):
             #   plt.imshow(edges[i], 'gray')
            #plt.show()
            


class Casia(Iris):
    _path = ''

    #def __init__(self, path=_path):
     #   self._path = path
        #self.load_image(self.path)

    def load_image(self, path=_path):

        # criar listas de lados, sujeitos e imagens
        self.subjects = []
        self.side = []
        self.images = []

        # percorrer as pastas
        subject_paths = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        for d in subject_paths:
            side_path = [os.path.join(d, s) for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]

            for s in side_path:
                for f in os.listdir(s):

                    if f.endswith('.jpg') and os.path.isfile(os.path.join(s,f)):


                        image_path = os.path.join(s, f)
                        image_pil = Image.open(image_path).convert('L')
                        image = np.array(image_pil, 'uint8')
                        side = os.path.split(s)[1]
                        subject = os.path.split(d)[1]


                        self.subjects.append(subject)
                        self.side.append(side)
                        self.images.append(image)





path = './CASIA-Iris-Lamp'
cassia = Casia(path)
#cassia.load_image(path)
cassia.preprocess()