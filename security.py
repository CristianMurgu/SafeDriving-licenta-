import string as str
import os
import shutil
import time
import zipfile
from turtledemo.penrose import start
from cryptography.fernet import Fernet
from dotenv import load_dotenv, set_key, dotenv_values

def create_key():
    key = Fernet.generate_key()
    return key.decode()

def write_to_env_usb(text):
    print(text)
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    if not os.path.exists(env_path):
        with open(env_path, "w"): pass

    key = "ENCRYPT_VALUE"

    set_key(env_path, key, text)

    load_dotenv(env_path, override = True)

    #print(f"ENCRYPT_VALUE = {os.getenv(key)}")

def check_content(content):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    if not os.path.exists(env_path):
        return False
    load_dotenv(env_path, override=True)
    if os.getenv("ENCRYPT_VALUE") == content:
        return True
    return False

def upload_drivers(dst):
    folder = "all_drivers"
    folder_zip = "all_drivers.zip"

    folder_path = os.path.join(os.getcwd(), folder)
    path_zip = os.path.join(os.getcwd(), folder_zip)
    with zipfile.ZipFile(path_zip, "w", zipfile.ZIP_DEFLATED) as zip_w:
        for file in os.listdir(folder_path):
            if file.endswith(".jpg"):
                file_path = os.path.join(folder_path, file)
                verif = os.path.relpath(file_path, start=os.getcwd())
                zip_w.write(file_path, verif)

    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
    time.sleep(1)
    shutil.move(path_zip, os.path.join(dst, folder_zip))
    #os.remove(folder_zip)



def search_official_usb():
    drives = []
    for drive in str.ascii_uppercase:
        if os.path.exists(f"{drive}:\\"):
            drives.append(f"{drive}:\\")

    for drive in drives:
        code = os.path.join(drive, "code_com1205.txt")
        if os.path.exists(code):
            content = open(code, "r").read().strip()
            if(check_content(content)):
                upload_drivers(drive)
                return True
    return False


