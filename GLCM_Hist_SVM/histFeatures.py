#!coding=utf-8
from PIL import Image
import numpy as np
import math

def getHistfeatures(img_path):
    img = Image.open(img_path)
    img = img.convert('1')
    im_array=np.array(img,dtype=int)
    print("im_array.shape={}".format(im_array.shape))
    
    L = 2
    Ps=np.zeros(L)
    nk=np.zeros(L)
    col = img.size[0]
    row = img.size[1]
    n=row*col
    
    for i in range(row):
        for j in range(col):
            num=long(im_array[i,j])
            nk[num]=nk[num]+1
    
    mu = 0
    delta = 0
    smooth=0
    S=0
    energy=0
    entropy=0
    for k in range(L):
        Ps[k]=nk[k]/n;
        mu=mu+(k+1)*Ps[k]
        print("Ps[k]={}".format(Ps[k]))
    
    print("mu={}".format(mu))
    
    for k in range(L):
        delta=delta+((k+1-mu)**2)*Ps[k]
        
        S=S+((k+1-mu)**3)*Ps[k]
        
        energy=energy+(Ps[k])**2
        if Ps[k]==0:
            Ps[k]=0.00000001
        entropy=entropy-(Ps[k]*(math.log(Ps[k])))
    print("delta={}".format(delta))
    
    smooth=1-float(1/(1+delta**2))
    S=S/(delta**0.5)**3
    print("smooth={}".format(smooth))
    print("S={}".format(S))
    print("energy={}".format(energy))
    print("entropy={}".format(entropy))
    return mu,delta,smooth,S,energy,entropy
