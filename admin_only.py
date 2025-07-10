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

    load_dotenv(env_path, override=True)

    # print(f"ENCRYPT_VALUE = {os.getenv(key)}")

write_to_env_usb(create_key())

