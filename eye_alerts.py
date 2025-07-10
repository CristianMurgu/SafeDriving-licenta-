
import tkinter as screen_tk
import numpy as np
import cv2 as cv
import ctypes
import threading
import time
import winsound

class EyeAlerts:
    def __init__(self):
        self.consecutive_closed = 0
        self.consecutive_phone = 0

    def isClosed(self, stmt):
        if stmt:
            self.consecutive_closed += 1
            if self.consecutive_closed == 3:
                self.alert("Wake up!")
                self.consecutive_closed = 0

        else:
            self.consecutive_closed = 0

    def isLookingPhone(self, stmt):
        if stmt:
            self.consecutive_phone += 1
            if self.consecutive_phone == 3:
                self.alert("Stop using the phone!")
                self.consecutive_phone = 0
        else:
            self.consecutive_phone = 0

    def sound(self):
        while getattr(threading.currentThread(), "do_run", True):
            winsound.Beep(1000, 500)
            time.sleep(0.33)

    def alert(self, text):
        screen_info = screen_tk.Tk()
        screen_width = screen_info.winfo_screenwidth()
        screen_height = screen_info.winfo_screenheight()
        win_width = int(screen_width * 0.3)
        win_height = int(screen_height * 0.3)

        thread = threading.Thread(target=self.sound)
        thread.daemon = True
        thread.start()

        bg = np.zeros((win_height, win_width, 3), dtype=np.uint8)
        font = cv.FONT_ITALIC
        scale = 2
        color = (255, 255, 255)
        thick = 2
        size = cv.getTextSize(text, font, scale, thick)[0]
        posX = (int(screen_width * 0.3) - size[0]) // 4
        posY = (int(screen_width * 0.3) - size[1]) // 2
        cv.putText(bg, text, (int(posX), int(posY * 0.5)), font, scale, color, thick, cv.LINE_AA)
        #scale = 1
        cv.putText(bg, "press 0", (int(posX), int(posY * 1)), font, scale, color, thick,
                   cv.LINE_AA)
        cv.putText(bg, "if acknowledged", (int(posX), int(posY * 1.25)), font, scale, color, thick,
                   cv.LINE_AA)

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

        thread.do_run = False
        thread.join()
