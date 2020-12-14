from image_loader import ImageLoader
import matplotlib.pyplot as plt

img_loader = ImageLoader(dir_name="images/train/train")
img = img_loader.get_one_image(labels={"cat": 0, "dog": 1}, new_size=(100, 100))
print(img["target"])
gray_scaled = img_loader.rgb2gray(img["data"])
sobeled = img_loader.sobel_image(gray_scaled)
plt.imshow(sobeled)
plt.show()
