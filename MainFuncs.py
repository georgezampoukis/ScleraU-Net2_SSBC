import numpy as np
import cv2
import os



def ProcessPredictedMask(mask, kernel, width, height, threshold, smooth, passes):

    mask = np.squeeze(mask)
    mask = mask * 255
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)

    if smooth:

        for i in range(passes):
            mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
            mask = cv2.erode(mask, (kernel, kernel), iterations=1)

    mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]

    return mask


def ProcessFrameForPrediction(frame, width, height):

    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    frame = np.reshape(frame, (1, frame.shape[0], frame.shape[1], 3))
    frame = frame.astype(np.float32)
    frame = frame / 255

    return frame
