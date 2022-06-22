import pytest
import decode
import encode
import findMasks
import image_number_of_ships
import ship_noship
import readImageFromS3

# All functions test
# decode test
@pytest.mark.parametrize('mask_rle',[('123')])
def test_decode(mask_rle):
    assert decode.rle_decode(mask_rle) == "The input run-length string cannot be decode"

@pytest.mark.parametrize('mask_rle',[("530845 5 531613 5 532381 5 533149 6 533918 5 534686 5 535454 5 536222 5 536990 5 537758 6 538527 5 539295 5 540063 2")])
def test_decode(mask_rle):
    assert decode.rle_decode(mask_rle) != "The input run-length string cannot be decode"


# encode test
@pytest.mark.parametrize('ImageId',[("123"),("123.jpg")])
def test_encode(ImageId):
    assert encode.rle_encode(ImageId) == "No such key! Please enter a valid image name!"

@pytest.mark.parametrize('ImageId',[("000155de5.jpg")])
def test_encode(ImageId):
    assert encode.rle_encode(ImageId) != "No such key! Please enter a valid image name!"


# findMasks test
@pytest.mark.parametrize('ImageId',[('000155de5.jpg'), ('00003e153.jpg')])
def test_findMasks(ImageId):
    assert findMasks.img_and_masks(ImageId) != "No such key! Please enter a valid image name!"

@pytest.mark.parametrize('ImageId',[('123'), ('123.jpg')])
def test_findMasks(ImageId):
    assert findMasks.img_and_masks(ImageId) == "No such key! Please enter a valid image name!"


# image_number_of_ships test
@pytest.mark.parametrize('num',[(0), (1), (9), (15)])
def test_image_number_of_ships(num):
    assert image_number_of_ships.image_num_ships(num) == [(150000),(27104),(243),(66)]

@pytest.mark.parametrize('num',[(-1), (20), (6.2)])
def test_image_number_of_ships(num):
    assert image_number_of_ships.image_num_ships(num).startswith("Error") == True


# ship_noship test
@pytest.mark.parametrize('t',[('ship'),('noship')])
def test_ship_noship(t):
    assert ship_noship.search_ship(t) != "Please type in 'ship' or 'noship'."

@pytest.mark.parametrize('t',[('123'),('abcd')])
def test_ship_noship(t):
    assert ship_noship.search_ship(t) == "Please type in 'ship' or 'noship'."


# readImageFromS3 test
@pytest.mark.parametrize('ImageId',[('000155de5.jpg'), ('00003e153.jpg')])
def test_readImageFromS3(ImageId):
    assert readImageFromS3.readImage_S3(ImageId) != "No such key! Please enter a valid image name!"

@pytest.mark.parametrize('ImageId',[('123'), ('123.jpg')])
def test_readImageFromS3(ImageId):
    assert readImageFromS3.readImage_S3(ImageId) == "No such key! Please enter a valid image name!"


if __name__=='__main__':
     pytest.main()