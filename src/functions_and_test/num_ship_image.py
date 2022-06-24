import numpy as np
import pandas as pd
from PIL import Image
import boto3
from io import StringIO
import botocore


def num_ship_in_image(ImageId: str):
    
    ''' 
    Input: ImageId : A string that contains the file name of the image in dataset.
    Return: an integer represents how many ships are there in this image.
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

    client = boto3.client('s3', aws_access_key_id = aws_key_id,
            aws_secret_access_key = aws_key)
    csv_obj = client.get_object(Bucket = bucket_name, Key = object_key_csv)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    data = pd.read_csv(StringIO(csv_string))
    data['ships'] = data['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    unique_img_ids = data.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['is_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
    res = unique_img_ids.loc[unique_img_ids['ImageId'] == ImageId, ['ships']]
    res = np.array(res)
    res = int(res[0])
    return res

print(num_ship_in_image('0a40de97d.jpg'))