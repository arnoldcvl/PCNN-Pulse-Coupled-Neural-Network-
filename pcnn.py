import numpy as np
import cv2 as cv

Alpha_F = 0.1
Alpha_L = 1.0
Alpha_T = 1.0
V_F = 0.5
V_L = 0.2
V_T = 20.0
Num = 10
Beta = 0.1

W = [0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 0.5,]
W = np.array(W, np.float).reshape((3, 3))
M = W

def main(argv):
    default_file = 'sudoku.png'
    filename = argv[0] if len(argv) > 0 else default_file
    
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
    return -1

    dim = src.shape

    global F = np.zeros( dim, np.float)
    global L = np.zeros( dim, np.float)
    global Y = np.zeros( dim, np.float)
    global T = np.ones( dim, np.float)

    #normalize image
    S = cv.normalize(src.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    
    for cont in range(Num):
        #numpy.convolve(W, Y, mode='same')
        F = np.exp(-Alpha_F) * F + V_F * numpy.convolve(W, Y, mode='same') + S
        L = np.exp(-Alpha_L) * L + V_L * numpy.convolve(M, Y, mode='same')
        U = F * (1 + Beta * L)
        T = np.exp(-Alpha_T) * T + V_T * Y
        Y = (U>T).astype(np.float)
        
if __name__ == "__main__":
    main(sys.argv[1:])
