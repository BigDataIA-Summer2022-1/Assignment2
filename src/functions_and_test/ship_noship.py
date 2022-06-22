# Author: Zifeng 
# Date: Jun 19th, 2022

import numpy as np
import boto3
from PIL import Image
import pandas as pd
from io import StringIO
import random
import botocore

def search_ship(t: str):
    '''
    Input: type: A string 'ship' or 'noship'
    Output: One of the images' name with ship(s) or noship in our dataset if input is 'ship' or 'noship', respectively.
    '''

    # AWS Credentials
    aws_key_id = 'AKIA2ZQ35MMOGV7ZZ7PA'
    aws_key = 'BrLIKkkVD+kdOQRz4TLp70K0YXZNaBHt6NVcfF2k'
    bucket_name = 'airbus-detection-team-1-re'
    object_key_csv = 'assignment-1/train_ship_segmentations_v2.csv'

    client = boto3.client('s3', aws_access_key_id = aws_key_id,
            aws_secret_access_key = aws_key)
    csv_obj = client.get_object(Bucket = bucket_name, Key = object_key_csv)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data = pd.read_csv(StringIO(csv_string))
    # dataset info
    ships = data[~data.EncodedPixels.isna()].ImageId.unique()
    noships = data[data.EncodedPixels.isna()].ImageId.unique()
    # randomly select an image index
    ran_ships = random.randint(0, len(ships) - 1)
    ran_noships = random.randint(0, len(noships) - 1)

    # Reuse the read image from S3 bucket function
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

    # Error handling
    if t == "ship":
        return readImage_S3(ships[ran_ships])

    elif t == "noship":
        return readImage_S3(noships[ran_noships])

    else:
        return "Please type in 'ship' or 'noship'."