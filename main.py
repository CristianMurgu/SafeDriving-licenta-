from fontTools.merge.util import current_time

import setup #1-restore, 2-set_driver, full_setup
import imgprocessing
import eyescollector
import drivers
import closed_open_eye_model
#import down_oth_eye_model
import cv2 as cv
import numpy as np
import time
import eye_tracker
import adjust_camera
from eye_alerts import EyeAlerts


def realtime():
    drivers.identify_driver()
    adjust_camera.set_camera()

    eye_analyzer = EyeAlerts()
    camera = cv.VideoCapture(1)

    if camera.isOpened() is False:
        print("Camera malfunction, exiting...")
        exit()

    last_time = time.time()

    try:
        while True:
            return_status, photo = camera.read()

            if return_status is False:
                print("Error taking input from camera, exiting...")
                break

            if time.time() - last_time >= 1:
                photo_closed_open, photo_eye_direction = imgprocessing.process_photo(photo)

                eyes = eyescollector.extract_eyes_coord(photo_eye_direction)
                print(eyes)

                if eyes is not None:

                    eyes_closed = True
                    watching_phone = False
                    for eye in eyes:

                        if closed_open_eye_model.is_eye_opened(photo_closed_open[eye[1]:eye[3], eye[0]:eye[2]]):
                            '''cv.imshow("ochii din umbra", photo_closed_open[eye[1]:eye[3], eye[0]:eye[2]])
                            cv.waitKey(0)'''
                            eyes_closed = False
                            break

                    if not eyes_closed:
                        if eye_tracker.is_looking_down(photo_eye_direction):
                            watching_phone = True

                    #print(f"eyes closed: {eyes_closed}")
                    eye_analyzer.isClosed(eyes_closed)

                    '''if not eyes_closed:
                        print(f"watching phone: {watching_phone}")'''
                    eye_analyzer.isLookingPhone(watching_phone)

                    '''cv.imshow('eye', photo)
                    cv.waitKey(0)'''

                last_time = time.time()



    finally:
        camera.release()
        cv.destroyAllWindows()



realtime()


'''photos_processed = imgprocessing.full_processing()

photo_label = 'driver17.jpg'
photo = cv.imread(photo_label)
photo = cv.resize(photo, (800, 800))
photo_closed_open, photo_eye_direction = imgprocessing.process_photo(photo)
eyes = eyescollector.eyes_extract(photo_closed_open)





cv.imshow('eye', photo_eye_direction)
cv.waitKey(0)'''
'''
#print(closed_open_eye_model.is_eye_opened(eyes[0]))
#print(down_oth_eye_model.is_eye_down(trimmed_eye[1]))
#imgprocessing.morphological_transformations_special(trimmed_eye[1])




#print(photos_processed)

cv.imshow("og", photo_closed_open)
cv.waitKey(0)'''

'''eyes = eyescollector.eyesExtract(photos_processed[1])

trimmed_eye = []

for eye in eyes:
    trimmed_eye.append(photos_processed[1][eye[1]:eye[1]+eye[3], eye[0]:eye[0]+eye[2]])
'''
'''cv.imshow("og", trimmed_eye[0])
cv.waitKey(0)'''
