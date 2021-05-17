#!/usr/bin/env python
# coding: utf-8

# In[16]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as clr
img = cv2.imread('C:/Users/user/Documents/image_process_project/HW1/Bird feeding 3 low contrast.tif')


# In[17]:


# if img is not None:
#     # show this image by cv2.imshow
#     cv2.imshow('image',img)    
    
#     # call cv2.waitKey to process window messages
#     cv2.waitKey()
    
#     # destroy all windows
#     cv2.destroyAllWindows()
# else:
#     print('image file is not found')


# In[18]:


inputValue=[]
arctanValue=[]
fcn={}
for num in range(0,256,1):
    inputValue.append(num)

for num in inputValue:
    arctanNum=np.arctan((num-128)/32)    
#     outputNum=num+97
#     if(outputNum>255):
#         outputNum-=256
    arctanValue.append(arctanNum)
#     fcn[num]=outputNum
y_shift=-min(arctanValue)
slope=( 255-0 )/( max(arctanValue)-min(arctanValue) )

for i in range(0,256,1):
    outputNum=round(slope*(arctanValue[i]+y_shift))
    fcn[inputValue[i]]=outputNum
# print(fcn)


# In[19]:


outImg=img.copy()
row,col,channel=img.shape
for y in range(row):
    for x in range(col):
#         print(fcn[img[y,x,0]])
        outImg[y,x,0]=fcn[img[y,x,0]]
        outImg[y,x,1]=fcn[img[y,x,1]]
        outImg[y,x,2]=fcn[img[y,x,2]]


# In[ ]:


if img is not None:
    cv2.imshow("out",outImg) 
    cv2.imshow("original",img)
     # call cv2.waitKey to process window messages
    cv2.waitKey()
    
    # destroy all windows
    cv2.destroyAllWindows()


# In[ ]:




