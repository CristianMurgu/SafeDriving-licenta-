import cv2 as cv
import numpy as np
import mediapipe as mpl
#from osam.apis import non_maximum_suppression

def one_time_load_face_cascades():
    face_cascades = []
    face_haarcascades = ['detection_data/haarcascade_frontalface_default.xml',
                         'detection_data/haarcascade_frontalface_alt.xml',
                         'detection_data/haarcascade_frontalface_alt2.xml',
                         'detection_data/haarcascade_frontalface_alt_tree.xml']

    for face_haarcascade in face_haarcascades:
        face_cascade = cv.CascadeClassifier(face_haarcascade)
        face_cascades.append(face_cascade)

    return face_cascades

def one_time_load_eye_cascades():
    eye_cascades = []
    eye_haarcascades = ['detection_data/haarcascade_eye.xml', 'detection_data/haarcascade_eye_tree_eyeglasses.xml',
                        'detection_data/haarcascade_lefteye_2splits.xml',
                        'detection_data/haarcascade_righteye_2splits.xml']

    for eye_haarcascade in eye_haarcascades:
        eye_cascade = cv.CascadeClassifier(eye_haarcascade)
        eye_cascades.append(eye_cascade)

    return eye_cascades

#setup for one time load haarcascades + mediapipe
face_cascades = one_time_load_face_cascades()
eye_cascades = one_time_load_eye_cascades()
facial_mesh = mpl.solutions.face_mesh
facial_mesh = facial_mesh.FaceMesh(static_image_mode = True, refine_landmarks = True, max_num_faces = 1, min_detection_confidence = 0.5)

def get_eyes_mediapipe(img):
    every_eye = []

    if len(img.shape) == 2:  # gray from checker
        height, width = img.shape
        img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    else:
        height, width, _ = img.shape
        img_rgb = img.copy()

    res = facial_mesh.process(img_rgb)
    if res.multi_face_landmarks is not None:
        for points in res.multi_face_landmarks:
            height, width, _ = img_rgb.shape
            landmarks_left_eye = [points.landmark[33], points.landmark[133], points.landmark[159], points.landmark[145],
                                  points.landmark[158], points.landmark[153]]
            landmarks_right_eye = [points.landmark[362], points.landmark[263], points.landmark[386],
                                   points.landmark[374], points.landmark[385], points.landmark[380]]

            min_x_left = min_y_left = min_x_right = min_y_right = np.inf
            max_x_left = max_y_left = max_x_right = max_y_right = 0

            for landmark in landmarks_left_eye:
                x, y = int(landmark.x * width), int(landmark.y * height)
                min_x_left = min(x, min_x_left)
                min_y_left = min(y, min_y_left)
                max_x_left = max(x, max_x_left)
                max_y_left = max(y, max_y_left)

            for landmark in landmarks_right_eye:
                x, y = int(landmark.x * width), int(landmark.y * height)
                min_x_right = min(x, min_x_right)
                min_y_right = min(y, min_y_right)
                max_x_right = max(x, max_x_right)
                max_y_right = max(y, max_y_right)

        if int(min_x_right * 0.95) >= 0:
            x1 = int(min_x_right * 0.95)
        else:
            x1 = min_x_right
        if int(min_y_right * 0.95) >= 0:
            y1 = int(min_y_right * 0.95)
        else:
            y1 = min_y_right
        if int((max_x_right - x1 * 0.9)) <= width:
            x2 = int((max_x_right - x1 * 0.9))
        else:
            x2 = int((max_x_right - x1))
        if int((max_y_right - y1 * 0.9)) <= height:
            y2 = int((max_y_right - y1 * 0.9))
        else:
            y2 = int((max_y_right - y1))

        every_eye.append((x1, y1, x2, y2))

        if int(min_x_left * 0.95) >= 0:
            x1 = int(min_x_left * 0.95)
        else:
            x1 = min_x_left
            print("da")
        if int(min_y_left * 0.95) >= 0:
            y1 = int(min_y_left * 0.95)
        else:
            y1 = min_y_left
            print("da")
        if int((max_x_left - x1 * 0.9)) <= width:
            x2 = int((max_x_left - x1 * 0.9))
        else:
            x2 = int((max_x_left - x1))
            print("da")
        if int((max_y_left - y1 * 0.9)) <= height:
            y2 = int((max_y_left - y1 * 0.9))
        else:
            y2 = int((max_y_left - y1))
            print("da")

        every_eye.append((x1, y1, x2, y2))
    return every_eye



def get_eyes_haarcascade(img):
    every_eye = []
    for eye_cascade in eye_cascades:
        eyes = eye_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=16
                                            , minSize=(35, 35), maxSize=(90, 90))
        if eyes is not None:
            every_eye.extend(eyes)

    non_maximum_suppression(every_eye)
    return every_eye



def non_maximum_suppression(paths, overlapThresh=0.5):
    if len(paths) == 0 : #or paths is None:
        return []

    areas = []
    for (x, y, w, h) in paths:
        areas.append((x, y, x + w, y + h))

    areas = np.array(areas)

    x1 = areas[:, 0]
    y1 = areas[:, 1]
    x2 = areas[:, 2]
    y2 = areas[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(y2)

    picks = []
    while len(idxs) > 0:
        highest = len(idxs) - 1
        y2_current = idxs[highest]

        picks.append(y2_current)

        x1_left_overlap = np.maximum(x1[y2_current], x1[idxs[:highest]])
        y1_top_overlap = np.maximum(y1[y2_current], y1[idxs[:highest]])
        x2_right_overlap = np.minimum(x2[y2_current], x2[idxs[:highest]])
        y2_bottom_overlap = np.minimum(y2[y2_current], y2[idxs[:highest]])

        width_intersection = np.maximum(0, x2_right_overlap - x1_left_overlap + 1)
        height_intersection = np.maximum(0, y2_bottom_overlap - y1_top_overlap + 1)

        overlap_ratio = (width_intersection * height_intersection) / area[idxs[:highest]]

        idxs = np.delete(idxs, np.concatenate(([highest], np.where(overlap_ratio > overlapThresh)[0])))

    return [areas[i] for i in picks]


def face_top_45(img):
    x_start_face, y_start_face, width_face, height_face = extract_biggest_face(img)
    face_path = img[y_start_face:y_start_face + int(height_face / 100 * 60), #first 45% of face containeyes
                   x_start_face:x_start_face + width_face]
    return face_path



'''def extract_eyes_path(img):
    if check_if_face(img):
        face_path = face_top_45(img)
    else:
        return None
    every_eye = get_eyes_mediapipe(img)
    driver_eyes = non_maximum_suppression(every_eye)
    extracted_driver_eyes = []
    for (ex, ey, ex2, ey2) in driver_eyes:
        extracted_driver_eyes.append(img[ey:ey2, ex:ex2])

    return extracted_driver_eyes'''


def extract_eyes_coord(img):
    if check_if_face(img):
        face_path = face_top_45(img)
    else:
        return None
    every_eye = get_eyes_mediapipe(img)
    driver_eyes = non_maximum_suppression(every_eye)

    return driver_eyes


def check_if_face(photo):
    every_face = []
    for face_cascade in face_cascades:
        faces = face_cascade.detectMultiScale(photo, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))
        if faces is not None:
            every_face.extend(faces)
    individual_faces = non_maximum_suppression(every_face)
    if len(individual_faces) == 0:
        return False
    return True

def extract_biggest_face(photo):
    every_face = []
    for face_cascade in face_cascades:
        faces = face_cascade.detectMultiScale(photo, scaleFactor=1.05, minNeighbors=5, minSize=(50, 50))
        if faces is not None:
            every_face.extend(faces)
    individual_faces = non_maximum_suppression(every_face)
    if len(individual_faces) == 0:
        return None
    individual_faces = np.array(individual_faces)
    keys = individual_faces[:, 3] * individual_faces[:, 2]
    sort_index = np.argsort(keys)[::-1]
    sorted_faces = []
    sorted_faces = np.array(sorted_faces)
    sorted_faces = individual_faces[sort_index]

    return sorted_faces[0]



def extract_best_img(photos):
    photo_biggest_eyes_avg = 999999999
    best_photo = None
    is_face = False
    index = -1

    for i, photo in enumerate(photos):
        #check for a full face and minimum 2 eyes
        if check_if_face(photo):
            is_face = True
            photo = face_top_45(photo)
            eyes_check1 = get_eyes_haarcascade(photo)
            eyes_check2 = get_eyes_mediapipe(photo)
            if len(eyes_check1) >= 1 and len(eyes_check2) ==2:
                    return photos[i], i

    if is_face:
        return None, "Eyes not visible"
    else:
        return None, "No face recognized"


