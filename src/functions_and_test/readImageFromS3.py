# Author: Zifeng 
# Date: Jun 19th, 2022

import numpy as np 
from PIL import Image
import boto3
import botocore

def readImage_S3(ImageId: str):
        '''
        This function's purpose is to read the image from AWS S3 Bucket to the memory.
        Input: Any image file name in the dataset.
        Output: The numpy array of the image pixles.
        '''

        # AWS Credentials
        aws_key_id = 'AKIA2ZQ35MMOGV7ZZ7PA'
        aws_key = 'BrLIKkkVD+kdOQRz4TLp70K0YXZNaBHt6NVcfF2k'
        bucket_name = 'airbus-detection-team-1-re'
        object_key_img = 'assignment-1/train_v2/' + ImageId

        # Error handling
        client = boto3.client('s3', aws_access_key_id = aws_key_id,
            aws_secret_access_key = aws_key)
        try:
            img_obj = client.get_object(Bucket = bucket_name, Key = object_key_img)
        except botocore.exceptions.ClientError:
            return "No such key! Please enter a valid image name!"
        body = img_obj['Body']
        img = Image.open(body)
        # create the numpy arrry of the image
        image_array = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
        return image_array