import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
def scatter3D (img, img_ori, vx, vy , vz):
    h_r, s_g, l_b = cv2.split(img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    
    pixel_colors = img_ori.reshape((np.shape(img_ori)[0]*np.shape(img_ori)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(h_r.flatten(), s_g.flatten(), l_b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel(vx)
    axis.set_ylabel(vy)
    axis.set_zlabel(vz)
    return fig

def crear_mascara (img):
    # Get the L channel
    Schannel = img[:,:,2]
    #change 250 to lower numbers to include more values as "white"
    mask = cv2.inRange(Schannel, 50, 255)

    #result = cv2.bitwise_and(img, img, mask=mask)
    return mask

def modificar_saturacion (rgb_img, mask):
    result = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2HLS)
    result[:,:,2] = result[:,:,2]*0.001
    result = cv2.cvtColor(result, cv2.COLOR_HLS2RGB)
    return result

def quitar_grilla (img, hls):
    mask= crear_mascara(hls)
    mask_inv = cv2.bitwise_not(mask)
    img_fin = modificar_saturacion(img,mask)
    img1_bg = cv2.bitwise_and(img,img,mask = mask_inv)
    img2_fg = cv2.bitwise_and(img_fin,img_fin,mask = mask)
    out_img = cv2.add(img1_bg,img2_fg)
    return out_img