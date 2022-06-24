import pathlib
import string
import time
import numpy as np
from fastapi import FastAPI, Request
from PIL import Image
import numpy as np
import boto3
from skimage.io import imread
import pandas as pd
from PIL import Image
import boto3
from io import StringIO
import botocore
import random
import json
import logging
import logging.config
import logging.handlers
from fastapi import FastAPI
import uvicorn


logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

logger = logging.getLogger(__name__)

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    idem = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"rid={idem} start request path={request.url.path}")
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    formatted_process_time = '{0:.2f}'.format(process_time)
    logger.info(f"rid={idem} completed_in={formatted_process_time}ms status_code={response.status_code}")
    
    return response


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
        return {"Error Messages: ": "The input run-length string cannot be decode"}
    
    return {"Image Pixel Array: ": str(img.reshape(shape))}

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
        return {"Error Messages: ": "Error! " + str(num) + " is not an integer between 0-" + str(m) + "."}
    # count the results
    count = 0
    for i in unique_img_ids['ships']:
        if i == num:
            count += 1
    return {"Number of Images: ": count}



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
            return {"Error Messages: ": "No such key! Please enter a valid image name!"}
        body = img_obj['Body']
        img = Image.open(body)
        # create the numpy arrry of the image
        image_array = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
        return {"Image Pixel Array: ": str(image_array)}

    # Error handling
    if t == "ship":
        return readImage_S3(ships[ran_ships])

    elif t == "noship":
        return readImage_S3(noships[ran_noships])

    else:
        return {"Error Messages: ": "Please type in 'ship' or 'noship'."}



@app.get("/readImage_S3/{ImageId}")
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
            return {"Error Messages: ": "No such key! Please enter a valid image name!"}
        body = img_obj['Body']
        img = Image.open(body)
        # create the numpy arrry of the image
        image_array = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
        return {"Image Pixel Array: ": str(image_array)}




@app.get("/image_and_masks/{ImageId}")
def img_and_masks(ImageId: str):
    ''' 
    Input: ImageId : A string that contains the file name of the image in dataset; ImgShape was set to default as 768 by 768
    Return: img: a numpy array represents the original image, all_masks: a numpy array represents the mask of the image
    If the name of the iamge file is invalid, return "No such key! Please enter a valid image name!"
    '''
    ImgShape = [768, 768]
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
        return {"Error Messages: ":"No such key! Please enter a valid image name!"}
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
    def rle_decode(mask_rle: str):
        '''
        Input: mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        '''
        shape = [768, 768]
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
            all_masks += rle_decode(mask).T
    
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
    return {"Image Pixel Array:":str(image_array), "Image Mask Array:":str(all_masks)}


@app.get("/num_ship_iamge/{ImageId}")
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
        return {"Error Messages: ": "No such key! Please enter a valid image name!"}

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
    return {"The number of ships in this image is: ": res}

if __name__ == "__main__":
    cwd = pathlib.Path(__file__).parent.resolve()
    uvicorn.run(app, host="127.0.0.1", port=4000, log_config=f"{cwd}/logging.conf")