#Q5------------------------------------------------------------------------
import cv2
from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np

originalImage = cv2.imread('/content/fingerprint.tif',0)
flipVertical = cv2.flip(originalImage, 0)
flipHorizontal = cv2.flip(originalImage, 1)
flipBoth = cv2.flip(originalImage, -1)
fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(originalImage, cmap="gray")
axs[0, 0].set_title('original Image')
axs[0, 1].imshow(flipVertical, cmap="gray")
axs[0, 1].set_title('flip Vertical')
axs[1, 0].imshow(flipHorizontal, cmap="gray")
axs[1, 0].set_title('flip Horizontal')
axs[1, 1].imshow(flipBoth , cmap="gray")
axs[1, 1].set_title('flip Both')

plt.subplots_adjust(bottom = 0.025 , hspace =0.5)
# histogram -------------------------------------------------------------------------------------------
hist = cv2.calcHist([originalImage],[0],None,[256],[0,256]) #images, channels, mask, histSize, ranges[, hist[, accumulate]]
plt.figure(figsize=(12, 5))
plt.hist(hist,np.arange(0,257,8) ,color =['lime'],rwidth = 0.5)
plt.xlabel('bins')
plt.ylabel('number of pixels')  
plt.xticks(np.arange(0,257,8))
plt.show()

# printing information ----------------------------------------------------------------------------------------------
shape = originalImage.shape 
print(" Height of photo : " +str(shape[0])+"\n Width of photo :"+str(shape[1])+"\n Number of pixels :"+str(shape[1]*shape[0]))

print(' Image size {}'.format(originalImage.size)) 
print(' Maximum RGB value in this image {}'.format(originalImage.max())) 
print(' Minimum RGB value in this image {}'.format(originalImage.min()))


print(" Pixel in [100,50] location contains : "+str(originalImage[100, 50]))
print(' Data type of a pixel {}'.format(originalImage[ 100, 50].dtype))

size = sys.getsizeof(originalImage)
print(' Image size is : ' + str(size)+' (Bytes) ')
print(' so we need ' + str(size/1024)+' kibibyte (KiB) of memory if we want to save it ( size / 1024)')
