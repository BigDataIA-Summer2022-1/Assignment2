# Author: Zifeng 
# Date: Jun 19th, 2022

from skimage.io import imread
import numpy as np
import pandas as pd
from PIL import Image
import boto3
from io import StringIO
import botocore
from matplotlib import pyplot as plt


def img_and_masks(ImageId: str, ImgShape = (768, 768)):
    ''' 
    Input: ImageId : A string that contains the file name of the image in dataset; ImgShape was set to default as 768 by 768
    Return: img: a numpy array represents the original image, all_masks: a numpy array represents the mask of the image
    If the name of the iamge file is invalid, return "No such key! Please enter a valid image name!"
    '''

    # AWS Credentials
    aws_key_id = 'AKIA2ZQ35MMOGV7ZZ7PA'
    aws_key = 'BrLIKkkVD+kdOQRz4TLp70K0YXZNaBHt6NVcfF2k'
    bucket_name = 'airbus-detection-team-1-re'
    object_key_img = 'assignment-1/train_v2/' + ImageId
    object_key_csv = 'assignment-1/train_ship_segmentations_v2.csv'

    # Error handling
    
    client = boto3.client('s3', aws_access_key_id = aws_key_id,
            aws_secret_access_key = aws_key)
    try:
        img_obj = client.get_object(Bucket = bucket_name, Key = object_key_img)
    except botocore.exceptions.ClientError:
        return "No such key! Please enter a valid image name!"
    body = img_obj['Body']
    img = Image.open(body)
    # create the numpy array of the image
    image_array = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)

    # Read the csv file from S3
    client = boto3.client('s3', aws_access_key_id = aws_key_id,
            aws_secret_access_key = aws_key)
    csv_obj = client.get_object(Bucket = bucket_name, Key = object_key_csv)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    masks = pd.read_csv(StringIO(csv_string))
    #num_masks = masks.shape[0]
    #print('number of training images', num_masks)
    img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros(ImgShape)

    # Reuse the decode function
    def rle_decode(mask_rle: str, shape):
        '''
        Input: mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)

    for mask in img_masks:
        # Note that NaN should compare as not equal to itself
        if mask == mask:
            all_masks += rle_decode(mask, ImgShape).T
    
    #fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
    #axarr[0].axis('off')
    #axarr[1].axis('off')
    #axarr[2].axis('off')
    #axarr[0].imshow(img)
    #axarr[1].imshow(all_masks)
    #axarr[2].imshow(img)
    #axarr[2].imshow(all_masks, alpha=0.4)
    #plt.tight_layout(h_pad=0.1, w_pad=0.1)
    #plt.show()
    return image_array, all_masks
    

def main(ImageId: str="d5d4183a0.jpg"):
    data = img_and_masks(ImageId)
    return data


if __name__ == "__main__":
    main()


print(img_and_masks("d5d4183a0.jpg"))