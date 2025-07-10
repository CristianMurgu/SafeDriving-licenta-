import cv2 as cv
import os
import glob
import time
import tkinter as screen
import ctypes

class CustomError(Exception):
    pass

#restore app to starting point after being used
#prev driver is deleted from module's memory
#TESTED
def restore():
    prog_path = os.getcwd()
    jpg_files = glob.glob(os.path.join(prog_path, '*.jpg'))

    for jpg_file in jpg_files:
        try:
            os.remove(jpg_file)
            print("App started normally")
        except Exception as ex:
            print(f"Error deleting {jpg_file}: {ex}")



#open camera for reference photo of driver (true-next step)
def set_driver():
    done = False

    screen_info = screen.Tk()
    screen_width = screen_info.winfo_screenwidth()
    screen_height = screen_info.winfo_screenheight()
    screen_info.destroy()

    win_width = int(screen_width * 0.6)
    win_height = int(screen_height * 0.6)

    x_window = int((screen_width - win_width) / 2)
    y_window = int((screen_height - win_height) / 2)

    driver_recog_cam = cv.VideoCapture(1)
    if not driver_recog_cam.isOpened():
        raise IOError("Camera module malfunctioned")
    cv.namedWindow("Set driver", cv.WINDOW_NORMAL)

    hwnd = ctypes.windll.user32.FindWindowW(None, "Set driver")
    GWL_STYLE = -16
    WS_VISIBLE = 0x10000000
    WS_BORDER = 0x00800000
    new_style = WS_VISIBLE | WS_BORDER
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, new_style)

    cv.resizeWindow("Set driver", win_width, win_height)
    cv.moveWindow("Set driver", x_window, y_window)

    start_time = time.time()
    photos_taken = 0
    while photos_taken<4:
        spent_time = int(time.time() - start_time)

        photo_taken_status, photo = driver_recog_cam.read()
        display_text = photo.copy()
        cv.putText(display_text, f"Look ahead {photos_taken+1}/4 photos ", (50, 50),
                   cv.FONT_ITALIC, 0.7, (0, 0 ,0), 2, cv.LINE_AA )
        cv.imshow("Set driver", display_text)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if (spent_time % 3 == 0 and photos_taken != 0) or spent_time == 4:
            photo_path = os.path.join(os.getcwd(), f"driver{photos_taken}.jpg")
            if not photo_taken_status:
                raise CustomError("Photo not taken correctly")
            else:
                cv.imwrite(photo_path, photo)
                print(f"photo{photos_taken} of driver successful set")
                done = True
                photos_taken += 1
            time.sleep(1)
    driver_recog_cam.release()
    cv.destroyWindow("Set driver")
    return done

def full_setup():
    restore()
    return set_driver()



