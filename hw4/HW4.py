#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import math

img = cv2.imread('C:/Users/user/Documents/image_process_project/HW4/Car On Mountain Road.tif',cv2.IMREAD_GRAYSCALE)
row,col=img.shape

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


n=21 #kern_size
sigma=3.5

h_n=(n-1)/2
x, y = np.meshgrid(np.arange(-h_n, h_n+1), np.arange(-h_n, h_n+1))
a=(x**2 + y**2 - (2.0*sigma**2)) / sigma**4
b= np.exp( - (x**2 + y**2) / (2*sigma**2) )
kernel=a*b

# normal = 1 / (2.0 * np.pi * sigma**2)
# kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * np.exp(-(x**2+y**2) / (2.0*sigma**2))# / normal
log = np.zeros_like(img, dtype=float)
# print(x)


# In[4]:


nkernel=kernel-np.mean(kernel)
nlog_figure=np.zeros_like(nkernel,dtype=float)
slope_value=255/(nkernel.max()-nkernel.min())
eigen_y=nkernel.min()   #y-eigen_y=slope_value(x-eigen_x),eigen_x=0
nlog_figure=slope_value*( nkernel.copy() )+eigen_y
nlog_figure=nlog_figure.astype(np.uint8,copy=False)
nlog_figure=cv2.resize(nlog_figure,(420,420))
cv2.imshow('nlog_figure',nlog_figure)
cv2.waitKey(0)
cv2.destroyAllWindows()
# for i in range(img.shape[0]-(kern_size-1)):
#     for j in range(img.shape[1]-(kern_size-1)):
#         window = img[i:i+kern_size, j:j+kern_size] * kernel
#         LoG[i, j] = np.sum(window)
# nLoG = LoG - np.mean(LoG)
# LoG = LoG.astype(np.int64, copy=False)
# print(nkernel.shape)


# In[5]:


zero_img=img.copy()
zero_img=np.pad(array=zero_img, pad_width=((math.floor(n/2),math.floor(n/2)),(math.floor(n/2),math.floor(n/2))), mode='constant', constant_values=0)
proc_img=zero_img.copy()

for i in range(row-n+1):
    for j in range(col-n+1):
        window = img[i:i+n, j:j+n] * nkernel
        log[i, j] = np.sum(window)
# for i in range(math.floor(n/2),math.floor(n/2)+row):
#     for j in range(math.floor(n/2),math.floor(n/2)+col):
#         window = zero_img[i-math.floor(n/2):i + math.floor(n/2)+1, j-math.floor(n/2):j + math.floor(n/2)+1] * nkernel
#         log[i-math.floor(n/2), j-math.floor(n/2)] = np.sum(window)


# In[6]:


log = log.astype(np.int64, copy=False)
# log_img=log.copy()
# log_img=log_img.astype(np.uint8, copy=False)
# cv2.imshow('log_img',log_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[7]:


# zero_img=img.copy()
# zero_img=np.pad(array=zero_img, pad_width=((math.floor(n/2),math.floor(n/2)),(math.floor(n/2),math.floor(n/2))), mode='constant', constant_values=0)
# proc_img=img.copy()
# for i in range( math.floor(n/2)+1, math.floor(n/2)+row-(n-1),1):
#     for j in range( math.floor(n/2)+1, math.floor(n/2)+col-(n-1),1):
#         proc_img[i-math.floor(n/2),j-math.floor(n/2)]=np.sum(  zero_img[i-math.floor(n/2):i+math.floor(n/2), j-math.floor(n/2):j+math.floor(n/2)]* nLoG  )
# # print(proc_img)


# In[8]:


# out = proc_img - np.min(proc_img)
# out = 255 * (out / np.max(out))
# cv2.imshow('zeroed',out)
# cv2.imshow('process',proc_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[9]:


threshold=0
threshold1=0.04
# zct_image=np.zeros_like(log)
out_zct_0=np.zeros_like(log)
out_zct=np.zeros_like(log)


# In[10]:


for i in range(2,row-1,1):
    for j in range(2,col-1,1):        
        if ((log[i,j+1]>=0 and log[i,j-1]<0) or (log[i,j+1]<0 and log[i,j-1]>=0)):
#             zct_image[i,j]=log[i,j]
            out_zct_0[i,j] = 255 * (abs(log[i,j+1]-log[i,j-1])> log.max()*threshold)
            out_zct[i,j] = 255*(abs(log[i,j+1]-log[i,j-1])> log.max()*threshold1)
        elif((log[i+1,j]>=0 and log[i-1,j]<0) or (log[i+1,j]<0 and log[i-1,j]>=0) ):
#             zct_image[i,j]=log[i,j]
            out_zct_0[i,j] =255*(abs(log[i+1,j]-log[i-1,j])>log.max()*threshold) 
            out_zct[i,j] = 255*(abs(log[i+1,j]-log[i-1,j])>log.max()*threshold1)
        elif( (log[i+1,j+1]>=0 and log[i-1,j-1]<0) or (log[i+1,j+1]<0 and log[i-1,j-1]>=0) ):
#             zct_image[i,j]=log[i,j]
            out_zct_0[i,j] = 255*(abs(log[i+1,j+1]-log[i-1,j-1])>log.max()*threshold)
            out_zct[i,j] = 255*(abs(log[i+1,j+1]-log[i-1,j-1])>log.max()*threshold1)
        elif( (log[i+1,j-1]>=0 and log[i-1,j+1]<0) or (log[i+1,j-1]<0 and log[i-1,j+1]>=0)):
#             zct_image[i,j]=log[i,j]
            out_zct_0[i,j] = 255*(abs(log[i+1,j-1]-log[i-1,j+1])>log.max()*threshold)
            out_zct[i,j] = 255*(abs(log[i+1,j-1]-log[i-1,j+1])>log.max()*threshold1)
# print(out_zct_0)


# In[11]:


out_zct_0=out_zct_0.astype(np.uint8, copy=False)
out_zct=out_zct.astype(np.uint8, copy=False)
cv2.imshow('0%',out_zct_0)
cv2.imshow('4%',out_zct)
cv2.waitKey(0) 
cv2.destroyAllWindows()
cv2.imwrite('0%.png',out_zct_0)
cv2.imwrite('4%.png',out_zct)


# In[12]:


lines = cv2.HoughLines(out_zct,1,np.pi/180,200)


# In[ ]:




