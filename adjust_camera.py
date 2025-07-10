import cv2 as cv
import tkinter as screen
import ctypes

import imgprocessing
import mediapipe as mpl

def set_camera():
    camera = cv.VideoCapture(1)

    screen_info = screen.Tk()
    screen_width = screen_info.winfo_screenwidth()
    screen_height = screen_info.winfo_screenheight()
    screen_info.destroy()

    win_width = int(screen_width * 0.6)
    win_height = int(screen_height * 0.6)

    x_window = int((screen_width - win_width) / 2)
    y_window = int((screen_height - win_height) / 2)

    cv.namedWindow("Adjust camera height", cv.WINDOW_NORMAL)

    hwnd = ctypes.windll.user32.FindWindowW(None, "Adjust camera height")
    GWL_STYLE = -16
    WS_VISIBLE = 0x10000000
    WS_BORDER = 0x00800000
    new_style = WS_VISIBLE | WS_BORDER
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, new_style)
    ctypes.windll.user32.ShowWindow(hwnd, 0)

    cv.resizeWindow("Adjust camera height", win_width, win_height)
    cv.moveWindow("Adjust camera height", x_window, y_window)

    if camera.isOpened() is False:
        return False
    is_set = False
    is_display_on = False
    facial_mesh = mpl.solutions.face_mesh
    facial_mesh = facial_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    is_not_alligned = True
    while is_not_alligned:
        return_status, img = camera.read()
        y_nose = 0
        if not return_status:
            continue
        aux_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = facial_mesh.process(aux_img)
        height, width, _ = img.shape
        if res.multi_face_landmarks:
            for face in res.multi_face_landmarks:
                nose = face.landmark[4]

                y_nose = int(nose.y * height)
        if (height / 100 * 40) <= y_nose <= (height / 100 * 60):
            is_not_alligned = False
            camera.release()
            break
            return None
        else:
            ctypes.windll.user32.ShowWindow(hwnd, 5)
            while camera.isOpened():
                return_status, img = camera.read()
                if not return_status:
                    continue
                aux_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                res = facial_mesh.process(aux_img)
                cv.line(img, (0, int(height / 2)), (width, int(height / 2)), (0, 255, 255), 3)
                if res.multi_face_landmarks:
                    for face in res.multi_face_landmarks:
                        nose = face.landmark[4]
                        x_nose = int(nose.x * width)
                        y_nose = int(nose.y * height)
                        cv.circle(img, (x_nose, y_nose), 3, (255, 255, 255), -2)
                        cv.line(img, (x_nose, y_nose), (x_nose, int(height / 2)), (0, 0, 255), 1)
                        if (height / 100 * 45) <= y_nose <= (height / 100 * 55):
                            text = "Aligned, press q"
                            font = cv.FONT_ITALIC
                            font_scale = 0.7
                            thickness = 2
                            pos = (50, 50)

                            text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
                            text_width, text_height = text_size

                            bg_text_x1 = int(pos[0] * 0.9)
                            bg_text_y1 =int((pos[1] - text_height) * 0.9)
                            bg_text_x2 = int((pos[0] + text_width))
                            bg_text_y2 = int((pos[1] + int(text_height * 0.2)) * 1.1)

                            cv.rectangle(img, (bg_text_x1, bg_text_y1), (bg_text_x2, bg_text_y2), (0, 0, 0), -1)
                            cv.putText(img, text, (50, 50),
                                   font, font_scale, (0, 255, 255), thickness, cv.LINE_AA)
                cv.imshow("Adjust camera height", img)

                #if (height / 100 * 40) >= y_nose <= (height / 100 * 60):
                if cv.waitKey(1) & 0xFF == ord('q') and (height / 100 * 40) <= y_nose <= (height / 100 * 60):
                    break
            #camera.release()
            cv.destroyWindow("Adjust camera height")
    return None

