import requests
import numpy as np
from PIL import Image

url = "http://127.0.0.1:5000/predict"

def test_image(color, name):
    img = Image.fromarray(np.full((224, 224, 3), color, dtype=np.uint8))
    img.save(name)
    with open(name, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    print(f"{name} ->", response.json())

test_image([255, 0, 0], "red.jpg")
test_image([0, 255, 0], "green.jpg")
test_image([0, 0, 255], "blue.jpg")
test_image([0, 0, 0], "black.jpg")
test_image([255, 255, 255], "white.jpg")
