import numpy as np
from scipy import signal

def del2(im):
    [ylen, xlen] = im.shape
    im_new = np.zeros([ylen, xlen], dtype=np.float32)
    for j in range(1, ylen-1):
        for i in range(1, xlen-1):
            im_new[j,i] = (im[j-1,i]+im[j+1,i]+im[j,i-1]+im[j,i+1])/4-im[j,i]
    return im_new


def srad(im, delta):
    q0 = 1
    for n in range(1, 6):
        [ylen, xlen] = im.shape
        X = np.zeros([ylen+2, xlen+2], dtype=np.float32)
        X[1:ylen+1, 1:xlen+1] = im
        # padding
        X[0, 1:xlen+1] = im[0, :]
        X[ylen+1, 1:xlen+1] = im[ylen-1, :]
        X[:, 0] = X[:, 1]
        X[:, xlen+1] = X[:, xlen]

        q0 = q0*np.exp(-delta)
        gRx = signal.convolve2d(X, [[0,0,0],[0,1,-1],[0,0,0]], mode='same', boundary='symm')
        gRy = signal.convolve2d(X, [[0,-1,0],[0,1,0],[0,0,0]], mode='same', boundary='symm')
        gLx = signal.convolve2d(X, [[0,0,0],[1,-1,0],[0,0,0]], mode='same', boundary='symm')
        gLy = signal.convolve2d(X, [[0,0,0],[0,-1,0],[0,1,0]], mode='same', boundary='symm')
        q1 = np.sqrt(gRx*gRx+gRy*gRy+gLx*gLx+gLy*gLy)/(X+0.0001)
        q2 = 4*del2(X)/(X+0.0001)        
        q = np.sqrt((1/2*(q1*q1)-1/16*(q2*q2))/((1+1/4*q2)*(1+1/4*q2)+0.01)) 
        c = 1/(1+((q*q-q0*q0)/(q0*q0*(1+q0*q0))))
        d = signal.convolve2d(c, [[0,0,0],[0,0,-1],[0,0,0]],  mode='same', boundary='symm')* \
            signal.convolve2d(X, [[0,0,0],[0,1,-1],[0,0,0]], mode='same', boundary='symm')+ \
            signal.convolve2d(c, [[0,0,0],[0,-1,0],[0,0,0]],  mode='same', boundary='symm')* \
            signal.convolve2d(X, [[0,0,0],[-1,1,0],[0,0,0]], mode='same', boundary='symm')+ \
            signal.convolve2d(c, [[0,-1,0],[0,0,0],[0,0,0]],  mode='same', boundary='symm')* \
            signal.convolve2d(X, [[0,-1,0],[0,1,0],[0,0,0]], mode='same', boundary='symm')+ \
            signal.convolve2d(c, [[0,0,0],[0,-1,0],[0,0,0]],  mode='same', boundary='symm')* \
            signal.convolve2d(X, [[0,0,0],[0,1,0],[0,-1,0]], mode='same', boundary='symm')
        X = X+delta/4*d
        im = X[1:ylen+1, 1:ylen+1]
    return im

def dicomp(im1, im2):
    im1 = srad(im1, 0.15)
    im2 = srad(im2, 0.15)
    im_di = abs(np.log((im1+1)/(im2+1)))
    im_di = srad(im_di, 0.15)
    return im_di

