import shutil

from PIL import Image
import requests
import os
import hashlib

# using requests - an HTTP library
def get(url, path="images/"):
    # requests.get is an HTTP get request that returns an "response" object
    r = requests.get(url, stream=True, allow_redirects=True, timeout=30)
    r.raise_for_status()
    # Preventing the downloaded image’s size from being zero, by forcing it to decompress
    r.raw.decode_content = True

    path = os.path.join(path, hashlib.sha1(url.encode()).hexdigest()[:5]+".jpg")
    print(path)
    with Image.open(r.raw) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(path)
    r.close()

def download(url_file, path="images/"):
    os.makedirs(path, exist_ok=True)
    with open(url_file, "r") as f:
        for url in f.read().split("\n"):
            try:
                get(url, path=path)
            except Exception as e:
                print(e)
    print ("k")

# remove all files from folder after inference is done
def remove(path="images/"):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

#download("../brightness_model/assets/dark.txt")
#download("../brightness_model/assets/bright.txt")
#remove()
