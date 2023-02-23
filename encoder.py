import cv2
import numpy as np

source = cv2.imread("source.png").astype('int32')
hide = cv2.imread("hide.png").astype('int32')

sh, sw, sc = source.shape
HEIGHT, WIDTH, CHANNEL = hide.shape

size_arr = np.array([HEIGHT, WIDTH, CHANNEL])
size_bin = (((size_arr[:,None] & (1 << np.arange(32)))) > 0).astype('int32')[:,::-1].flatten()

new_hide = hide.flatten()
hide_bin = (((new_hide[:,None] & (1 << np.arange(8)))) > 0).astype('int32')[:,::-1].flatten()

msg = np.concatenate((size_bin, hide_bin))

source = source.flatten()

source[:len(msg)] = source[:len(msg)] + (source[:len(msg)] % 2 != msg).astype('int32')
source[:len(msg)] = source[:len(msg)] - (source[:len(msg)] == 256).astype('int32') * 2

source = source.reshape(sh, sw, sc)
cv2.imwrite("encoded_image.png", source)