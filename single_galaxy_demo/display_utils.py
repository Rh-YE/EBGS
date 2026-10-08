"""Original default DESI RGB display, extracted from main.py."""
import numpy as np

def arcsinh_rgb(imgs, mode="CHW", m = 0.03, clip=True):
    bands = ["g", "r", "z"]
    rgbscales =dict(g=(2,6.0), r=(1,3.4), z=(0,2.2))

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    Q = 50
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-9
    H,W = I.shape
    if mode == "HWC":
        rgb = np.zeros((H,W,3), np.float32)
        for img,band in zip(imgs, bands):
            plane,scale = rgbscales[band]
            rgb[:,:,plane] = (img * scale + m) * fI / I
    elif mode == "CHW":
        rgb = np.zeros((3,H,W), np.float32)
        for img,band in zip(imgs, bands):
            plane,scale = rgbscales[band]
            rgb[plane] = (img * scale + m) * fI / I
    if clip:
        rgb = np.clip(rgb, 0, 1)
    return rgb
