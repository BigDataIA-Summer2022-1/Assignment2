# Author: Zifeng 
# Date: Jun 19th, 2022

from PIL import Image
import numpy as np
import boto3

def rle_encode(ImageId: str):
    '''
    Input: ImageId: image file name
    Return: run length encoding as string formated
    '''

    # AWS Credentials
    aws_key_id = 'AKIA2ZQ35MMOGV7ZZ7PA'
    aws_key = 'BrLIKkkVD+kdOQRz4TLp70K0YXZNaBHt6NVcfF2k'
    bucket_name = 'airbus-detection-team-1-re'
    object_key_img = 'assignment-1/train_v2/' + ImageId

    # Error handling
    try:
        client = boto3.client('s3', aws_access_key_id = aws_key_id,
            aws_secret_access_key = aws_key)
    except:
        return "No such key! Please enter a valid image name!" 
    img_obj = client.get_object(Bucket = bucket_name, Key = object_key_img)
    body = img_obj['Body']
    img = Image.open(body)
    # create the numpy arrry of the image
    image_array = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)

    pixels = image_array.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main(ImageId: str="d5d4183a0.jpg"):
    data = rle_encode(ImageId)
    return data
if __name__ == "__main__":
    main()
