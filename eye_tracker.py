import mediapipe as mpl
import cv2 as cv
from sympy.physics.units import current
import numpy as np

import eyescollector
import imgprocessing




facial_mesh = mpl.solutions.face_mesh

facial_mesh = facial_mesh.FaceMesh(static_image_mode = True, refine_landmarks = True, max_num_faces = 1, min_detection_confidence = 0.5)

adjustments_needed = 0
eye_height_contour = 0

def isnt_retina_visible(photo, x, y):
    r, g, b = photo[y, x]
    initial_lum = 0.299 * r + 0.587 * g + 0.114 * b

    px = 0
    for y_aux in range(y - 3, y + 4):
        for x_aux in range(x - 3, x + 4):
            if 0 > y_aux >= photo.shape[0] or 0 > x_aux >= photo.shape[1]:
                continue
            else:
                r, g, b = photo[y_aux, x_aux]
                current_lum = 0.299 * r + 0.587 * g + 0.114 * b
                if current_lum * 1.5 >= initial_lum:
                    px += 1
    print(px)
    if px > 15:
        return True
    return False

def is_looking_down(photo):
    '''photo_label = "driver13.jpg"
    photo = cv.imread(photo_label)'''
    '''photo = imgprocessing.morphological_transformations_bgr(photo)
    photo = cv.cvtColor(photo, cv.COLOR_BGR2RGB)'''

    #photo = cv.resize(photo, (800, 800))
    #res = facial_mesh.process(photo)

    '''if eyescollector.check_if_face(photo):
        photo = eyescollector.face_top_45(photo)'''


    res = facial_mesh.process(photo)
    eyes = eyescollector.extract_eyes_coord(photo)
    global adjustments_needed, eye_height_contour

    eyes_down = 0
    special_case = False

    if eyes:


        '''for (ex, ey, ex1, ey1) in eyes:
            ex_start = min(ex_start, ex)
            ex_end = max(ex_end, ex1)
            ey_start = min(ey_start, ey)
            ey_end = max(ey_end, ey1)
            if res.multi_face_landmarks is not None:
                for points in res.multi_face_landmarks:
                    height, width, _ = photo.shape
                    landmarks = [points.landmark[468], points.landmark[473]]
                    for landmark in landmarks:
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        cv.circle(photo, (x, y), 3, (0, 255, 255), 1)
                        if ey_start < y < ey_end:
                            print("schameeeee")'''
        for (ex, ey, ex1, ey1) in eyes:
            #cv.rectangle(photo, (ex, ey), (ex1, ey1), (0, 0, 0), 6)
            if res.multi_face_landmarks is not None:
                for points in res.multi_face_landmarks:
                    height, width, _ = photo.shape
                    landmarks = [points.landmark[468], points.landmark[473]]
                    landmarks_left = [points.landmark[159], points.landmark[145]]
                    landmarks_right = [points.landmark[386], points.landmark[374]]

                    if (eye_height_contour >= 1.5 * abs(landmarks_left[0].y * height - landmarks_left[1].y * height) or eye_height_contour >= 1.5 * abs(landmarks_right[0].y * height - landmarks_right[1].y * height)) and adjustments_needed >= 5:
                        special_case = True

                    eye_height_contour = max(eye_height_contour, abs(landmarks_left[0].y * height - landmarks_left[1].y * height))
                    eye_height_contour = max(eye_height_contour, abs(landmarks_right[0].y * height - landmarks_right[1].y * height))

                    if adjustments_needed <= 5:
                        adjustments_needed += 1

                    #print(f"daa + {adjustments_needed}")

                    '''for landmark in landmarks_left:
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        cv.circle(photo,(x, y), 3, (255, 255, 255), 1)

                    for landmark in landmarks_right:
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        cv.circle(photo,(x, y), 3, (255, 255, 255), 1)
                    '''
                    for landmark in landmarks:
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        #cv.circle(photo,(x, y), 3, (0, 255, 255), 1)
                        #print(ex, x, ex1)
                        #print(ey, y, ey1)
                        if ex <= x <= ex1 and ey <= y <= ey1:
                            #print(f"var1 {y}")
                            #print(f"var2 {(ey1 + ey) / 2}")
                            if y > (ey + ey1) / 2: #or isnt_retina_visible(photo, x, y):
                                eyes_down+=1
        #cv.rectangle(photo, (ex_start, ey_start), (ex_end, ey_end), (0, 0, 0), 6)
    '''cv.imshow("testing", photo)
    cv.waitKey(0)'''
    if eyes_down >= 1 or special_case:
        return True
    return False


'''photo_label = "driver22.jpg"
photo = cv.imread(photo_label)

print(is_looking_down(photo))'''

#todo check on the mid point each eye (radius to be det, if it has only retina in the radius