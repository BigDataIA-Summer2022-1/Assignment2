# Author: Zifeng 
# Date: Jun 19th, 2022

import numpy as np

def rle_decode(mask_rle: str, shape = (768, 768)):
    '''
    Input: mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    try:
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    except:
        return "The input run-length string cannot be decode"
    return img.reshape(shape)


    