#!/usr/bin/env mdl
# -*- coding:utf-8 -*-
import math
import cv2
import numpy as np
import random


MEAN_FACE = np.array([
    [-0.17607, -0.172844],  # left eye pupil
    [0.1736, -0.17356],  # right eye pupil
    [-0.00182, 0.0357164],  # nose tip
    [-0.14617, 0.20185],  # left mouth corner
    [0.14496, 0.19943],  # right mouth corner
])


def get_mean_face(mf, face_width, canvas_size):
    ratio = face_width / (canvas_size * 0.34967)
    left_eye_pupil_y = mf[0][1]
    # In an aligned face image, the ratio between the vertical distances from eye to the top and bottom is 1:1.42
    ratioy = (left_eye_pupil_y * ratio + 0.5) * (1 + 1.42)
    mf[:, 0] = (mf[:, 0] * ratio + 0.5) * canvas_size
    mf[:, 1] = (mf[:, 1] * ratio + 0.5) * canvas_size / ratioy

    return mf


def get_align_transform(lm, mf):
    mx = mf[:, 0].mean()
    my = mf[:, 1].mean()
    dmx = lm[:, 0].mean()
    dmy = lm[:, 1].mean()

    ux = mf[:, 0] - mx
    uy = mf[:, 1] - my
    dux = lm[:, 0] - dmx
    duy = lm[:, 1] - dmy
    c1 = (ux * dux + uy * duy).sum()
    c2 = (ux * duy - uy * dux).sum()
    c3 = (dux**2 + duy**2).sum()
    a = c1 / c3
    b = c2 / c3

    kx, ky = 1, 1

    transform = np.zeros((2, 3))
    transform[0][0] = kx * a
    transform[0][1] = kx * b
    transform[0][2] = mx - kx * a * dmx - kx * b * dmy
    transform[1][0] = -ky * b
    transform[1][1] = ky * a
    transform[1][2] = my - ky * a * dmy + ky * b * dmx
    return transform


def align_5p(*imgs, ld, face_width, canvas_size, translation=[0, 0], rotation=0, scale=1, sa=1, sb=1):
    nose_tip = ld[30]
    left_eye = np.mean(ld[36:42], axis=0).astype('int')
    right_eye = np.mean(ld[42:48], axis=0).astype('int')
    left_mouth, right_mouth = ld[48], ld[54]
        
    lm = np.array([left_eye, right_eye, nose_tip, left_mouth, right_mouth])  # ld

    mf = MEAN_FACE * scale
    mf = get_mean_face(mf, face_width, canvas_size)

    M1 = np.eye(3)
    M1[:2] = get_align_transform(lm, mf)

    M2 = np.eye(3)
    M2[:2] = cv2.getRotationMatrix2D((canvas_size/2, canvas_size/2), rotation, 1)

    def stretch(va, vb, s):
        m = (va + vb) * 0.5
        d = (va - vb) * 0.5
        va[:] = m + d * s
        vb[:] = m - d * s

    mf = mf[[0, 1, 3, 4]].astype(np.float32)
    mf2 = mf.copy()
    stretch(mf2[0], mf2[1], sa)
    stretch(mf2[2], mf2[3], 1.0/sa)
    stretch(mf2[0], mf2[2], sb)
    stretch(mf2[1], mf2[3], 1.0/sb)

    # import ipdb; ipdb.set_trace()
    mf2 += np.array(translation)

    M3 = cv2.getPerspectiveTransform(mf, mf2)
    # import ipdb; ipdb.set_trace()

    M = M3.dot(M2).dot(M1)

    # np.array(translation)[np.newaxis]

    dshape = (canvas_size, canvas_size)
    return [cv2.warpPerspective(img, M, dshape) for img in imgs]


def rand_range(rng, lo, hi):
    return rng.rand()*(hi-lo)+lo


def jpeg_encode(img, quality=90):
    '''
    :param img: uint8 color image array
    :type img: :class:`numpy.ndarray`
    :param int quality: quality for JPEG compression
    :return: encoded image data
    '''
    return cv2.imencode('.jpg', img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tostring()


def _clip_normalize(img):
    return np.clip(img, 0, 255).astype('uint8')


def adjust_gamma(img, gamma):
    k = 1.0 / gamma
    img = cv2.exp(k * cv2.log(img.astype('float32') + 1e-15))
    f = 255.0 ** (1 - k)
    return _clip_normalize(img * f)


def get_linear_motion_kernel(angle, length):
    """:param angle: in degree"""
    rad = np.deg2rad(angle)

    dx = np.cos(rad)
    dy = np.sin(rad)
    a = int(max(list(map(abs, (dx, dy)))) * length * 2)
    if a <= 0:
        return None

    kern = np.zeros((a, a))
    cx, cy = a // 2, a // 2
    dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
    cv2.line(kern, (cx, cy), (dx, dy), 1.0)

    s = kern.sum()
    if s == 0:
        kern[cx, cy] = 1.0
    else:
        kern /= s

    return kern


def linear_motion_blur(img, angle, length):
    kern = get_linear_motion_kernel(angle, length)
    return cv2.filter2D(img, -1, kern)


def gaussian_noise(rng, img, sigma):
    """add gaussian noise of given sigma to image"""
    return _clip_normalize(img + rng.randn(*img.shape) * sigma)


def adjust_tone(src, color, p):
    dst = ((1 - p) * src + p * np.ones_like(src) * np.array(color).reshape((1, 1, len(color))))
    return _clip_normalize(dst)


_CV2_RESIZE_INTERPOLATIONS = [
    cv2.INTER_CUBIC,
    cv2.INTER_LINEAR,
    cv2.INTER_NEAREST,
    cv2.INTER_AREA,
    cv2.INTER_LANCZOS4
]


def resize_rand_interp(rng, img, size):
    return cv2.resize(
        img, size, interpolation=rng.choice(_CV2_RESIZE_INTERPOLATIONS))


def add_noise(rng, img):
    # apply jpeg_encode augmentor
    if rng.rand() > 0.7:
        b_img = jpeg_encode(img, quality=int(15 + rng.rand() * 65))
        img = cv2.imdecode(np.fromstring(b_img, np.uint8), cv2.IMREAD_UNCHANGED)

    # do normalize first
    if rng.rand() > 0.5:
        img = (img - img.min()) / (img.max() - img.min() + 1) * 255

    # quantization noise
    if rng.rand() > .5:
        ih, iw = img.shape[:2]
        noise = rng.randn(ih//4, iw//4) * 2
        noise = cv2.resize(noise, (iw, ih))
        img = np.clip(img + noise[:, :, np.newaxis], 0, 255)

    # apply HSV augmentor
    if rng.rand() > 0.75:
        img = np.array(img, 'uint8')
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if rng.rand() > 0.5:
            if rng.rand() > 0.5:
                r = 1. - 0.5 * rng.rand()
            else:
                r = 1. + 0.15 * rng.rand()
            hsv_img[:, :, 1] = np.array(np.clip(hsv_img[:, :, 1] * r, 0, 255), 'uint8')
        if rng.rand() > 0.5:
            # brightness
            if rng.rand() > 0.5:
                r = 1. + rng.rand()
            else:
                r = 1. - 0.5 * rng.rand()
            hsv_img[:, :, 2] = np.array(np.clip(hsv_img[:, :, 2] * r, 0, 255), 'uint8')
        img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    img = adjust_gamma(img, (0.6 + rng.rand() * 0.8))

    if rng.rand() > 0.7:  # motion blur
        r_angle = int(rng.rand() * 360)
        r_len = int(rng.rand() * 10) + 1
        img = linear_motion_blur(img, r_angle, r_len)
    if rng.rand() > 0.7:
        img = cv2.GaussianBlur(img, (3, 3), rng.randint(3))
    if rng.rand() > 0.7:
        if rng.rand() > 0.5:
            img = gaussian_noise(rng, img, rng.randint(15, 22))
        else:
            img = gaussian_noise(rng, img, rng.randint(0, 5))

    if rng.rand() > 0.7:
        # append color tone adjustment
        rand_color = tuple([60 + 195 * rng.rand() for _ in range(3)])
        img = adjust_tone(img, rand_color, rng.rand()*0.3)

    # apply interpolation
    x, y = img.shape[:2]
    if rng.rand() > 0.75:
        r_ratio = rng.rand() + 1  # 1~2
        target_shape = (int(x / r_ratio), int(y / r_ratio))
        resize_rand_interp(rng, img, target_shape)
        resize_rand_interp(rng, img, (x, y))

    return np.array(img, 'uint8')


def resize_aug(img, ld):
    resize_ratio = 2
    rh, rw = img.shape[:2]
    for i in range(len(ld)):
        ld[i] = [ld[i][0]/resize_ratio, ld[i][1]/resize_ratio]
    img = cv2.resize(img, (rw//resize_ratio, rh//resize_ratio), interpolation=cv2.INTER_LINEAR)
    return img, ld


if __name__ == "__main__":
    img = cv2.imread("./images/manipulated_sequences/Deepfakes/raw/frames/035_036/frame_0.png")
    rng = np.random
    rng.seed(999)
    img = add_noise(rng, img)
    cv2.imwrite("tmp.png", img)
# vim: ts=4 sw=4 sts=4 expandtab
