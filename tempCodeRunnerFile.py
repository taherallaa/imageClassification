import numpy as np
from PIL import Image

# Load the image and resize it to 100x100
img = Image.open("data2/cat.0.jpg").resize((100, 100))

# Convert the image to grayscale and normalize the pixel values
img_array = np.array(img.convert('L')) / 255.0

# Apply max pooling with a pool size of (2, 2)
pooled_array = np.zeros((50, 50))
for i in range(50):
    for j in range(50):