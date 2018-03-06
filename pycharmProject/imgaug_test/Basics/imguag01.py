#coding:utf-8
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
'''
seq = iaa.Sequential([
    iaa.Crop(px=(0,16)),    #crop images from each side by 0 to 16px(randomly)
    iaa.Fliplr(0.5),    #flip the image from left to right
    iaa.GaussianBlur(sigma=(0,3.0)) #blur the image with a sigma of 0 to 3.0  blur operation and denoise the image
])

for batch_idx in  range(1000):
    #the 'images' should be either 4D numpy array of shape (N, height, width, channels)
    #or a list of 3D numpy arrays, each having shape(height, width, channels)
    #Grayscale images must have shape(height, width, 1) each.
    #All images must have numpy's dtype uint8.Values are expected to be in range 0 to 255
    images = load_batch(batch_idx)
    images = seq.augment_images(images)
    train_on_images(images_aug)
'''

from imgaug import augmenters as iaa
import numpy as np

images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 3.0))])

# show an image with 8*8 augmented versions of image 0
seq.show_grid(images[0], cols=8, rows=8)

# Show an image with 8*8 augmented versions of image 0 and 8*8 augmented
# versions of image 1. The identical augmentations will be applied to
# image 0 and 1.
seq.show_grid([images[0], images[1]], cols=8, rows=8)
