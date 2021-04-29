#!/usr/bin/env python
# coding: utf-8

# In[52]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as clr
img = cv2.imread('C:/Users/user/Documents/image_process_project/HW1/Bird feeding 3 low contrast.tif')


# In[53]:


# if img is not None:
#     # show this image by cv2.imshow
#     cv2.imshow('image',img)    
    
#     # call cv2.waitKey to process window messages
#     cv2.waitKey()
    
#     # destroy all windows
#     cv2.destroyAllWindows()
# else:
#     print('image file is not found')


# In[54]:


inputValue=[]
for num in range(0,256,1):
    inputValue.append(num)
outputValue=[]
fcn={}
for num in inputValue:
    outputNum=num+97
    if(outputNum>255):
        outputNum-=256
    outputValue.append(outputNum)
#     fcn[num]=outputNum


for i in range(0,256,1):
    fcn[inputValue[i]]=outputValue[i]
# print(fcn)


# In[55]:


outImg=img.copy()
row,col,channel=img.shape
for y in range(row):
    for x in range(col):
#         print(fcn[img[y,x,0]])
        outImg[y,x,0]=fcn[img[y,x,0]]
        outImg[y,x,1]=fcn[img[y,x,1]]
        outImg[y,x,2]=fcn[img[y,x,2]]


# In[56]:


if img is not None:
    cv2.imshow("out",outImg) 
    cv2.imshow("original",img)
     # call cv2.waitKey to process window messages
    cv2.waitKey()
    
    # destroy all windows
    cv2.destroyAllWindows()


# In[ ]:




