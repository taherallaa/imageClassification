import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
counter = 0
P = []
T = []
weights = np.array([])
bias = np.array([])
image = None
text = ""
path = ""

# ------------------------------------------------------------


def convert_img(img_path):
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    return get_feature(img_array)


def get_feature(image):
    new = conv_relu(image)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = conv_relu(new)
    new = pooling(new)
    new = flatten(new)

    return new


def conv_relu(image):
    mask = [[-1, -1, 1], [0, 1, -1], [0, 1, 1]]
    size1 = len(image) - 2
    size2 = len(image[0]) - 2
    new_image = [[0 for _ in range(size2)]for _ in range(size1)]
    for i in range(size1):
        for j in range(size2):
            x = 0
            for k in range(3):
                x += (image[i+k][j+0]*mask[k][0] + image[i+k][j+1]
                      * mask[k][1] + image[i+k][j+2]*mask[k][2])
            new_image[i][j] = x if x > 0 else 0

    return new_image


def pooling(image):
    size1 = int(len(image)/2)
    size2 = int(len(image[0])/2)
    new_image = [[0 for _ in range(size2)]for _ in range(size1)]
    for i in range(0, size1):
        for j in range(0, size2):
            x = 0
            for k in range(2):
                x += (image[(i*2)+k][(j*2)+0] + image[(i*2)+k][(j*2)+1])/4
            new_image[i][j] = int(x)
    return new_image


def flatten(image):
    new_image = []
    for row in image:
        for el in row:
            new_image.append(el)
    return new_image


def select_file():
    global counter, text, image, path
    file_path = filedialog.askopenfilename()
    if file_path:
        path = file_path
        print(f"Selected file: {file_path}")
        display_image(file_path)
        image = ImageTk.PhotoImage(Image.open(file_path))

        text = ""
        label_res.configure(text=text)


def display_image(file_path):
    image = Image.open(file_path)
    image = image.resize((300, 300))
    photo = ImageTk.PhotoImage(image)
    label_image.configure(image=photo)
    label_image.image = photo

# ------------------------------------------------------------


def classify():
    global image, weights, text, path
    p = convert_img(path)
    w = [[]]
    for i in weights:
        w[0].append(i)

    tmp = []
    for i in p:
        l = []
        l.append(i)
        tmp.append(l)

    w = np.array(w)
    tmp = np.array(tmp)

    print(w.shape)
    print(tmp.shape)

    n = np.dot(w, tmp)
    print(n[0][0])

    text = "Type is : Cat" if n[0][0] >= 0 else "Type is : Dog"

    label_res.configure(text=text)


# trian function to train model
def train():
    global weights, T, P
    for i in range(10):
        P.append(convert_img(f"data2/cat.{i}.jpg"))
        T.append(1)
        P.append(convert_img(f"data2/dog.{i}.jpg"))
        T.append(-1)
        print("i train ", str(i + 1))
    P = np.array(P)
    T = np.array(T)
    weights = np.dot(T, np.dot(np.linalg.inv(np.dot(P, P.T)), P))
    print("Trained done")
    
    
# ---------------------------------------------------------------


train()


# Run GUI
root = tk.Tk()
root.geometry("400x430")

frame_buttons = tk.Frame(root)
frame_buttons.pack(side=tk.BOTTOM, pady=10)

button_upload = tk.Button(
    frame_buttons, text="Upload Photo", command=select_file)
button_upload.pack(side=tk.LEFT, padx=10)

button_classify = tk.Button(frame_buttons, text="Classify", command=classify)
button_classify.pack(side=tk.LEFT, padx=10)

label_image = tk.Label(root)
label_image.pack()

label_res = tk.Label(root)
label_res.pack()

root.mainloop()
