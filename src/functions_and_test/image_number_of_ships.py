# Author: Zifeng 
# Date: Jun 19th, 2022

import pandas as pd 
import boto3
from io import StringIO

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