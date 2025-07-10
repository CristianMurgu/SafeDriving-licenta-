import glob
import cv2 as cv
import os
import sys
import setup
import imgprocessing
import eyescollector
import pyautogui as screen
import numpy as np
import tkinter as screen_tk
import ctypes

import security



def identify_driver():
    photo_current_driver = None
    auth_user = False
    while True:
        text = "photo_current_driver"
        if security.search_official_usb() == True:
            text = "files uploaded"
            auth_user = True
        else:
            setup.full_setup()
        photos_processed = imgprocessing.full_processing()
        photo_current_driver, index = eyescollector.extract_best_img(photos_processed)
        if type(index) is not int and auth_user == False:
            text = index
        if photo_current_driver is not None and auth_user == False:
            break

        setup.restore()

        screen_info = screen_tk.Tk()
        screen_width = screen_info.winfo_screenwidth()
        screen_height = screen_info.winfo_screenheight()
        win_width = int(screen_width * 0.3)
        win_height = int(screen_height * 0.3)

        bg = np.zeros((win_height, win_width, 3), dtype = np.uint8)
        font = cv.FONT_ITALIC
        scale = 2
        color = (255, 255, 255)
        thick = 2
        size = cv.getTextSize(text, font, scale, thick)[0]
        posX = (int(screen_width * 0.3) - size[0]) // 4
        posY = (int(screen_width * 0.3) - size[1]) // 2
        cv.putText(bg, text, (int(posX), int(posY * 0.5)), font, scale, color, thick, cv.LINE_AA)
        cv.putText(bg, "press 0", (int(posX), int(posY * 1)), font, scale, color, thick, cv.LINE_AA)
        cv.putText(bg, "when you're ready", (int(posX), int(posY * 1.25)), font, scale, color, thick,cv.LINE_AA)

        #win_width = int(screen_width * 0.3)
        #win_height = int(screen_height * 0.3)

        screen_info.destroy()

        x_window = int((screen_width - win_width) / 2)
        y_window = int((screen_height - win_height) / 2)

        cv.namedWindow("Warning", cv.WINDOW_NORMAL)
        cv.resizeWindow("Warning", win_width, win_height)
        cv.moveWindow("Warning", x_window, y_window)

        hwnd = ctypes.windll.user32.FindWindowW(None, "Warning")
        GWL_STYLE = -16
        WS_VISIBLE = 0x10000000
        WS_BORDER = 0x00800000
        new_style = WS_VISIBLE | WS_BORDER
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, new_style)

        cv.imshow("Warning", bg)
        cv.waitKey(0)
        cv.destroyWindow("Warning")
        if auth_user:
            sys.exit()

    store_driver(index)
    print(index)
    setup.restore()
    return photo_current_driver, index

def get_next_index(folder):
    return len(glob.glob(os.path.join(folder, '*.jpg')))

def store_driver(driver_index):
    folder = "all_drivers"
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.chmod(folder, 0o777) #admin-rwx group-rw, oth-r 0o764

    src = os.path.join(os.getcwd(), f"driver{driver_index}.jpg")
    print(src)
    img = cv.imread(src)
    new_index = get_next_index(folder)
    dst = os.path.join(folder, f"driver{new_index}.jpg")
    cv.imwrite(dst, img)
    print(dst)


'''cv.imshow("driver", identify_driver())
cv.waitKey(0)'''