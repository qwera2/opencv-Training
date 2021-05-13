#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
# import math

image = cv2.imread('C:/Users/user/Documents/image_process_project/HW3/Bird 3 blurred.tif')
row,col,channel=image.shape


# In[9]:


b=image[:, :, 0]
g=image[:, :, 1]
r=image[:, :, 2]
cv2.imshow('image',image)
cv2.imshow('B',b)
cv2.imshow('G', g)
cv2.imshow('R', r)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


b1=b/255
g1=g/255
r1=r/255


# In[11]:


hsi_image=image.copy()
# h,s,I=cv2.split(hsi_image)
# print(h)
h=0.0
s=0.0
I=0.0
for i in range(row):
    for j in range(col):
        numer=0.5*( (r1[i,j]-g1[i,j])+(r1[i,j]-b1[i,j]) )
        denom=np.sqrt( (r1[i,j]-g1[i,j])**2 + (r1[i,j]-b1[i,j])*(g1[i,j]-b1[i,j]) )
        
        theta = np.arccos(numer/denom)
#         if(i==j):
#             print(theta)
        
        if(denom<=0):
            h=0
        elif(b1[i,j]<=g1[i,j]):
            h=theta
        else:
            h=2*3.14159265-theta
            
#         print(h)
        
        min_RGB = min(min(b1[i, j], g1[i, j]), r1[i, j])
        suma = b1[i, j]+g1[i, j]+r1[i, j]
        if suma == 0:
            s= 0
        else:
            s= 1 - 3*min_RGB/suma
        
        h=h/(2*3.14159265)
        I=suma/3.0
        
        hsi_image[i,j,0]=h*255
        hsi_image[i,j,1]=s*255
        hsi_image[i,j,2]=I*255


# In[12]:


cv2.imshow('HSI',hsi_image)
h1,s1,i1=cv2.split(hsi_image)
cv2.imshow('H',h1)
cv2.imshow('S',s1)
cv2.imshow('I',i1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


diff_rgb_hsi_image=image.copy()
diff_rgb_hsi_image[:,:,0]=image[:,:,0]-hsi_image[:,:,0]
diff_rgb_hsi_image[:,:,0]=image[:,:,1]-hsi_image[:,:,1]
diff_rgb_hsi_image[:,:,0]=image[:,:,2]-hsi_image[:,:,2]

cv2.imshow('rgb-hsi',diff_rgb_hsi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




