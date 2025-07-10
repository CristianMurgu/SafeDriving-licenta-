
import cv2 as cv
import os
import numpy as np
import glob
import matplotlib.pyplot as plt

def process_photo(photo):
    photo = cv.resize(photo, (800, 800))
    photo_gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)

    '''clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    photo_clahe = clahe.apply(photo_gray)'''

    #gauss_blur = cv.GaussianBlur(photo_gray, (7, 7), 0)

    photo_gray = morphological_transformations_grayscale(photo_gray)
    '''_, photo_bin = cv.threshold(gauss_blur, 127, 255, cv.THRESH_BINARY)
    photo_bin = morphological_transformations_binary(photo_bin)'''
    photo_rgb = morphological_transformations_bgr(photo)
    photo_rgb = cv.cvtColor(photo, cv.COLOR_BGR2RGB)

    return photo_gray, photo_rgb


def get_photos():
    photos = []
    photos_labels = glob.glob(os.path.join(os.getcwd(), '*.jpg'))
    #print(photos_ids)
    for photo_label in photos_labels:
        photo = cv.imread(photo_label)
        photo = cv.resize(photo, (800, 800))

        photo_gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
        photo_gray = cv.equalizeHist(photo_gray)
        clahe = cv.createCLAHE(clipLimit= 2.0, tileGridSize=(10, 10))
        photo_clahe = clahe.apply(photo_gray)
        _, photo = cv.threshold(photo_clahe, 127, 255, cv.THRESH_BINARY)
        '''cv.imshow('daaa', photo)
        cv.waitKey(0)'''


        photos.append(photo_gray)
    return np.array(photos) , np.array(photos_labels)
    #print(photos)
    #cv.destroyAllWindows()


def fill_hole(img):
    img_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(img_floodfill, mask, (0,0), 255)
    img_floodfill_inv = cv.bitwise_not(img_floodfill)
    img_out = img | img_floodfill_inv
    return img_out


def morphological_transformations_grayscale(photo):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    photo_clahe = clahe.apply(photo)
    photo_black = cv.morphologyEx(photo, cv.MORPH_TOPHAT, kernel, iterations=1)
    photo_top = cv.morphologyEx(photo, cv.MORPH_BLACKHAT, kernel, iterations=1)

    photo_enh = cv.add(photo_clahe, photo_top)
    photo_enh = cv.subtract(photo_enh, photo_black)

    ''' photo = cv.dilate(photo, kernel, iterations= 1) #increase bright on darker spots
    photo = cv.morphologyEx(photo, cv.MORPH_OPEN, kernel, iterations = 1) #removing bright noises
    photo = cv.morphologyEx(photo, cv.MORPH_TOPHAT, kernel) #eyelash enhance for closed eye    #cum ba sa fie asta problema )=
    '''

    '''cv.imshow("daaaaaaaa", photo_enh)
    cv.waitKey(0)'''
    return photo_enh

def morphological_transformations_binary(photo):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    photo = cv.morphologyEx(photo, cv.MORPH_OPEN, kernel, iterations = 1)
    photo = cv.erode(photo, kernel, iterations = 1)


    #photo = cv.morphologyEx(photo, cv.MORPH_GRADIENT, kernel, iterations = 1)

    #photo = cv.morphologyEx(photo, cv.MORPH_BLACKHAT, kernel, iterations=1)
    photo = cv.dilate(photo, kernel, iterations=4)


    return photo

def morphological_transformations_bgr(photo):
    lab = cv.cvtColor(photo, cv.COLOR_BGR2LAB)
    lightness, green_to_red, blue_to_yellow = cv.split(lab)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

    lightness = clahe.apply(lightness)
    lightness = cv.morphologyEx(lightness, cv.MORPH_OPEN, kernel, iterations = 1)
    lightness = cv.morphologyEx(lightness, cv.MORPH_CLOSE, kernel, iterations = 1)
    #lightness = cv.dilate(lightness, kernel, iterations = 1)

    lab_mod = cv.merge((lightness, green_to_red, blue_to_yellow))
    photo_mod = cv.cvtColor(lab_mod, cv.COLOR_LAB2BGR)

    return photo_mod

def morphological_transformations(photos):
    #1-dilation / erosion
    erosion = [] #not the best gray - scade luminozitate
    dilatation = [] #first todo gray - luminoziatete
    closing =[]
    opening = [] #second todo gray - izolare obiecte

    white_black_hat = []
    hole_fill = []
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5)) #path de 5x5 pixeliu test, inainte 3x3
    '''for img in photos:
        dilation_erosion.append(cv.erode(img, kernel, iterations= 1))
    #erosion works better?'''
    #1-erosion (smal noise remove)
    for img in photos:
        erosion.append(cv.erode(img, kernel, iterations = 1))

    #2-dilatation (restore size eye)
    for img in erosion:
        dilatation.append(cv.dilate(img, kernel, iterations= 1))

    #3-closing (close holes)
    for img in dilatation:
        closing.append(cv.morphologyEx(img, cv.MORPH_CLOSE, kernel))

    #4-opening (remove small obj)
    for img in closing:
        opening.append(cv.morphologyEx(img, cv.MORPH_OPEN, kernel))


    '''#3-white/black hat (for enhance bright resolution/features within an image
    for img in closing:
        white_black_hat.append(cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel, iterations= 2))

    #4-hole fill
    for img in white_black_hat:
        hole_fill.append(fill_hole(img))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(white_black_hat[0], cmap='gray')
    plt.show()
    return hole_fill'''
    return opening

def morphological_transformations_special(photos):

    #gauss_blur = cv.GaussianBlur(photos, (5, 5), 0)
    _, bin_auto_otsu = cv.threshold(photos, 0, 255, cv.THRESH_BINARY)

    cv.imshow("da", bin_auto_otsu)
    cv.waitKey(0)

def full_processing():
    processed_photos, photos_label = get_photos()
    return morphological_transformations(processed_photos)
    #return processed_photos
