import sys
import numpy as np
import cv2 as cv
from scipy import signal

Alpha_F = 0.1
Alpha_L = 1.0
Alpha_T = 0.3

V_F = 0.5
V_L = 0.2
V_T = 20.0

Num = 10
Beta = 0.1

W = [0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 0.5,]
W = np.array(W, np.float).reshape((3, 3))
M = W

global F
global L
global Y
global T
global Y_AC

def main(argv):
    default_file = 'images/3.jpeg'
    filename = argv[0] if len(argv) > 0 else default_file
    
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    cv.imshow("Original", src)

    dim = src.shape

    F = np.zeros( dim, np.float)
    L = np.zeros( dim, np.float)
    Y = np.zeros( dim, np.float)
    T = np.ones( dim, np.float)
    Y_AC = np.zeros( dim, np.float)
    
    #normalize image
    S = cv.normalize(src.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    
    for cont in range(Num):
        #numpy.convolve(W, Y, mode='same')
        F = np.exp(-Alpha_F) * F + V_F * signal.convolve2d(Y, W, mode='same') + S
        L = np.exp(-Alpha_L) * L + V_L * signal.convolve2d(Y, M, mode='same')
        U = F * (1 + Beta * L)
        T = np.exp(-Alpha_T) * T + V_T * Y
        Y = (U>T).astype(np.float)
        Y_AC = Y_AC + Y
    
    cv.imshow("Result", Y)
    Y_AC = cv.normalize(Y_AC.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    cv.imshow("Result Acumulated", Y_AC)

    #Y = (Y*255).astype(np.uint8)
    cv.imwrite('result.jpg', Y*255)
    cv.imwrite('result Acumulated.jpg', Y_AC*255)
        
    cv.waitKey()
    return 0
        
if __name__ == "__main__":
    main(sys.argv[1:])
