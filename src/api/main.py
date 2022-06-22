from typing import Tuple, Union
from typing import List
import numpy as np
from fastapi import FastAPI,HTTPException
from PIL import Image
import numpy as np
import boto3
from skimage.io import imread
import pandas as pd
from PIL import Image
import boto3
from io import StringIO
import botocore
from matplotlib import pyplot as plt
import random

app = FastAPI()


@app.get("/run_length_decode/{mask_rle}")
def rle_decode(mask_rle: str):
    '''
    Input: mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    shape = [768, 768]
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
    return print(img.reshape(shape))

@app.get("/image_number_of_ships/{num}")
def image_num_ships(num: int):
    # input: An integer: number of ships in an image.
    # output: How many images in our dataset with this certain number of ships. Or Error.

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
    data['ships'] = data['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = data.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['is_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
    #print(f"Count of images with number (0, 1, 2 etc.) of ships \n{unique_img_ids['ships'].value_counts()}\n\n")
    m = max(unique_img_ids['ships'])
    # Error handling
    if num < 0 or num > m or type(num) != int:
        return "Error! " + str(num) + " is not an integer between 0-" + str(m) + "."
    # count the results
    count = 0
    for i in unique_img_ids['ships']:
        if i == num:
            count += 1
    return count



@app.get("/ship_nonship_image/{t}")
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
        return print(image_array)

    # Error handling
    if t == "ship":
        return readImage_S3(ships[ran_ships])

    elif t == "noship":
        return readImage_S3(noships[ran_noships])

    else:
        return "Please type in 'ship' or 'noship'."