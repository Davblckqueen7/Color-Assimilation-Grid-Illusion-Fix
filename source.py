import numpy as np
import cv2
from matplotlib import pyplot as plot
from tkinter import Label, Tk
from tkinter import filedialog
import functions as fn

root = Tk()
path = filedialog.askopenfilename(filetypes=[("Archivo imagen", 'png')])
ddepth = cv2.CV_16S
kernel_size = 3

img = cv2.imread(path,1)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

#fn.scatter3D(rgb_img, rgb_img,'red', 'green', 'blue')
#fn.scatter3D(hsl_img, rgb_img, 'hue', 'luminosity', 'saturation')

plot.subplot(231), plot.imshow(rgb_img),plot.title("Imagen convertida a RGB")
plot.xticks([]), plot.yticks([])
plot.subplot(232), plot.imshow(hsl_img),plot.title("Imagen convertida a HLS")
plot.xticks([]), plot.yticks([])
plot.subplot(233), plot.imshow(fn.crear_mascara(hsl_img), cmap="gray"),plot.title("Mascara")
plot.xticks([]), plot.yticks([])
plot.subplot(234), plot.imshow(fn.modificar_saturacion(rgb_img, fn.crear_mascara(hsl_img)), cmap="gray"),plot.title("Mascara infrasaturada")
plot.xticks([]), plot.yticks([])
plot.subplot(235), plot.imshow(fn.quitar_grilla(rgb_img, hsl_img)),plot.title("Imagen final")
plot.xticks([]), plot.yticks([])

plot.show()

