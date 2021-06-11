#!/usr/bin/env python
# coding: utf-8

# In[86]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import math

image = cv2.imread('C:/Users/user/Documents/image_process_project/HW3/Bird 3 blurred.tif')
row,col,channel=image.shape


# In[87]:


def hsitorgb(hsi_img):
    h = int(hsi_img.shape[0])
    w = int(hsi_img.shape[1])
    H, S, I = cv2.split(hsi_img)
    H = H / 255.0
    S = S / 255.0
    I = I / 255.0
    bgr_img = hsi_img.copy()
    B, G, R = cv2.split(bgr_img)
    for i in range(h):
        for j in range(w):
            if S[i, j] < 1e-6:
                R = I[i, j]
                G = I[i, j]
                B = I[i, j]
            else:
                H[i, j] *= 360
                if H[i, j] > 0 and H[i, j] <= 120:
                    B = I[i, j] * (1 - S[i, j])
                    R = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    G = 3 * I[i, j] - (R + B)
                elif H[i, j] > 120 and H[i, j] <= 240:
                    H[i, j] = H[i, j] - 120
                    R = I[i, j] * (1 - S[i, j])
                    G = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    B = 3 * I[i, j] - (R + G)
                elif H[i, j] > 240 and H[i, j] <= 360:
                    H[i, j] = H[i, j] - 240
                    G = I[i, j] * (1 - S[i, j])
                    B = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j]*math.pi/180)) / math.cos((60 - H[i, j])*math.pi/180))
                    R = 3 * I[i, j] - (G + B)
            bgr_img[i, j, 0] = B * 255
            bgr_img[i, j, 1] = G * 255
            bgr_img[i, j, 2] = R * 255
    return bgr_img


# In[88]:


b=image[:, :, 0]
g=image[:, :, 1]
r=image[:, :, 2]
cv2.imshow('image',image)
cv2.imshow('B',b)
cv2.imshow('G', g)
cv2.imshow('R', r)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[89]:


b1=b/255
g1=g/255
r1=r/255
hsi_image=np.zeros_like(image)


# In[90]:



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
            h=2*np.pi-theta
            
#         print(h)
        
        min_RGB = min(min(b1[i, j], g1[i, j]), r1[i, j])
        suma = b1[i, j]+g1[i, j]+r1[i, j]
        if suma == 0:
            s= 0
        else:
            s= 1 - 3*(min_RGB/suma)
        
        h=h/(2*np.pi)
        I=suma/3.0
        
        hsi_image[i,j,0]=h*255
        hsi_image[i,j,1]=s*255
        hsi_image[i,j,2]=I*255


# In[91]:


# cv2.imshow('HSI',hsi_image)
h1,s1,i1=cv2.split(hsi_image)
cv2.imshow('H',h1)
cv2.imshow('S',s1)
cv2.imshow('I',i1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[92]:


kernel1=np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])
kernel2=np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])
c=1
r_o=np.zeros((row+2,col+2), np.uint8)
g_o=np.zeros((row+2,col+2), np.uint8)
b_o=np.zeros((row+2,col+2), np.uint8)
i_o=np.zeros((row+2,col+2), np.uint8)

r_s=np.zeros((row,col), np.uint8)
g_s=np.zeros((row,col), np.uint8)
b_s=np.zeros((row,col), np.uint8)
i_s=np.zeros((row,col), np.uint8)

r_o[1:row+1,1:col+1]=r
g_o[1:row+1,1:col+1]=g
b_o[1:row+1,1:col+1]=b
i_o[1:row+1,1:col+1]=I

rgb_s=np.zeros_like(image)


# In[93]:


for i in range(1,row+1,1):
    for j in range(1,col+1,1):
#         r_s[i-1,j-1]=np.sum( r_o[i-1:i+2,j-1:j+2]*kernel1 )
#         g_s[i-1,j-1]=np.sum( g_o[i-1:i+2,j-1:j+2]*kernel1 )
#         b_s[i-1,j-1]=np.sum( b_o[i-1:i+2,j-1:j+2]*kernel1 )
        i_s[i-1,j-1]=np.sum( i_o[i-1:i+2,j-1:j+2]*kernel2 )
        
# rgb_s[:,:,0]=b_s+b
# rgb_s[:,:,1]=g_s+g
# rgb_s[:,:,2]=r_s+r
rgb_s[:,:,0]=cv2.filter2D(b, -1, kernel1)
rgb_s[:,:,1]=cv2.filter2D(g, -1, kernel1)
rgb_s[:,:,2]=cv2.filter2D(r, -1, kernel1)

cv2.imshow('rgb_sharpen',rgb_s)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[94]:


hsi_s=np.zeros_like(image)
hsi_s[:,:,0]=h1
hsi_s[:,:,1]=s1
hsi_s[:,:,2]=i1+i_s
hsi_rgb=hsitorgb(hsi_s)

cv2.imshow('hsi_img',hsi_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[95]:


b_rgbs,g_rgbs,r_rgbs=cv2.split(rgb_s)
b_hsis,g_hsis,r_hsis=cv2.split(hsi_rgb)

diff_b=-( b_rgbs-b_hsis )
diff_g=-( g_rgbs-g_hsis )
diff_r=-( r_rgbs-r_hsis )
diff_total=diff_b+diff_g+diff_r

cv2.imshow('hsi-rgb sharpen',diff_total)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[96]:


diff_b1= rgb_s[:,:,0]-hsi_rgb[:,:,0] 
diff_g1= rgb_s[:,:,1]-hsi_rgb[:,:,1] 
diff_r1= rgb_s[:,:,2]-hsi_rgb[:,:,2] 
diff_total1=diff_b+diff_g+diff_r

cv2.imshow('rgb-hsi sharpen',diff_total1)
cv2.waitKey(0)
cv2.destroyAllWindows()

