#!/usr/bin/env python
# coding: utf-8

# In[70]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.colors as clr
img = cv2.imread('C:/Users/user/Documents/image_process_project/HW2/Bird 2.tif',cv2.IMREAD_GRAYSCALE)


# In[71]:


f=np.fft.fft2(np.double(img))
shift_f=np.fft.fftshift(f)
F_log = np.log(1+np.abs(shift_f))
plt.subplot(1,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.subplot(1,2,2)
plt.imshow(F_log, cmap = 'gray')
plt.title('magnitude spectrum')
plt.show()


# In[72]:


row,col=img.shape
cen_row=row/2
cen_col=col/2

d=30
filter_in30=np.zeros((row,col),np.uint8)
filter_out30=np.zeros((row,col),np.uint8)
img_double=np.zeros((row,col),np.uint8)
img_double=np.double(img)


# In[73]:


for i in range(row):
    for j in range(col):
        if np.sqrt((i-cen_row)**2+(j-cen_col)**2) < d:
            filter_in30[i,j]=1
            filter_out30[i,j]=0
        else:
            filter_in30[i,j]=0
            filter_out30[i,j]=1
            
f_double=np.fft.fft2(img_double)
f_s_double=np.fft.fftshift(f_double)
output_in=filter_in30*f_s_double
output_out=filter_out30*f_s_double


# In[74]:


output_in=np.fft.ifftshift(output_in)
output_out=np.fft.ifftshift(output_out)

output_in=np.fft.ifft2(output_in)
output_out=np.fft.ifft2(output_out)

output_in30_real=np.real(output_in)
output_in30_real-=np.min(output_in30_real)
output_in30_real=output_in30_real/np.max(output_in30_real)*255
output_in30_real=np.uint8(output_in30_real)

output_out30_real=np.real(output_out)
output_out30_real-=np.min(output_out30_real)
output_out30_real=output_out30_real/np.max(output_out30_real)*255
output_out30_real=np.uint8(output_out30_real)

output_in30_abs=abs(output_in)
output_in30_abs-=np.min(output_in30_abs)
output_in30_abs=output_in30_abs/np.max(output_in30_abs)*255
output_in30_abs=np.uint8(output_in30_abs)

output_out30_abs=abs(output_out)
output_out30_abs-=np.min(output_out30_abs)
output_out30_abs=output_out30_abs/np.max(output_out30_abs)*255
output_out30_abs=np.uint8(output_out30_abs)


# In[75]:


plt.subplot(1,2,1)
plt.imshow(output_in30_real, cmap = 'gray')
plt.title('inside r=30. Take real part'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(output_in30_abs, cmap = 'gray')
plt.title('inside r=30. Take abs value'), plt.xticks([]), plt.yticks([])
plt.show()


# In[76]:


plt.subplot(1,2,1)
plt.imshow(output_out30_real, cmap = 'gray')
plt.title('outside r=30. Take real part'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(output_out30_abs, cmap = 'gray')
plt.title('outside r=30. Take abs value'), plt.xticks([]), plt.yticks([])
plt.show()

