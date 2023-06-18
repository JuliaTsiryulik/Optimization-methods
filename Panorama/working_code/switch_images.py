from numpy import *
from matplotlib.pyplot import *
import cv2 as cv
from PIL import Image

import warp


# example of affine warp of im1 onto im2
im2 = array(Image.open('data/test.jpg').convert('L'))
im1 = array(Image.open('data/fisherman.jpg').convert('L'))
# set to points

tp = array([[226,352,352,220],[60,98,192,162],[1,1,1,1]])
im3 = warp.image_in_image(im1,im2,tp)
figure()
gray()
imshow(im3)
axis('equal')
axis('off')
show()