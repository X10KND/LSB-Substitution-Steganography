import cv2
import numpy as np

encoded = cv2.imread("encoded_image.png").astype('int32')

encoded_arr = encoded.flatten()
enc_bin = encoded_arr % 2

HEIGHT = int("".join(str(x) for x in enc_bin[0:32]), 2)
WIDTH = int("".join(str(x) for x in enc_bin[32:64]), 2)
CHANNEL = int("".join(str(x) for x in enc_bin[64:96]), 2)

enc_bin = enc_bin[96 : 96 + HEIGHT * WIDTH * CHANNEL * 8].reshape(-1,8)
dec_arr = np.sum(enc_bin[:] * [128, 64, 32, 16, 8, 4, 2, 1], axis=1)

dec_arr = dec_arr.reshape(HEIGHT, WIDTH, CHANNEL)

cv2.imwrite("decoded_image.png", dec_arr)