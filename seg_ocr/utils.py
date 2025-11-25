#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, List, Dict
import numpy as np
import cv2
from PIL import Image


def pil_to_numpy(img_pil: Image.Image, code = cv2.COLOR_RGB2BGR) -> np.ndarray:
    """
    PIL.Image.open() → PIL.Image.Image（RGB，类对象）
    cv2.imread() → numpy.ndarray（BGR，数组）
    OpenCV 的函数只能吃 numpy.ndarray，所以不能直接把 PIL 对象丢进去。
    解决办法：用 np.array(img_pil) 转换成数组，再 cv2.cvtColor() 转换颜色顺序。

    Args:
        img_pil (Image): _description_

    Returns:
        np.ndarray: _description_
    """
    # 读入图像（PIL）
    # img_pil = Image.open("test.png")

    # 转换为 numpy 数组
    img_np = np.array(img_pil)

    # 注意：PIL 是 RGB，而 OpenCV 是 BGR
    img_bgr = cv2.cvtColor(img_np, code=code)

    return img_bgr


def numpy_to_pil(img_numpy: np.ndarray, code = cv2.COLOR_BGR2RGB) -> Image.Image:
    img_rgb = cv2.cvtColor(img_numpy, code=code)
    img_pil_out = Image.fromarray(img_rgb)
    return img_pil_out



def sanitize_for_ocr(img: Union[str, np.ndarray]):
    # 允许输入是路径或 ndarray
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)  # 可能读出灰度/4通道
        if img is None:
            raise ValueError(f"Failed to read image: {img}")

    if img is None:
        raise ValueError("Input image is None")

    # 保证是 ndarray
    img = np.asarray(img)

    # 尺寸检查
    if img.ndim == 2:
        # 灰度 -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        ch = img.shape[2]
        if ch == 4:
            # BGRA -> BGR（丢弃 alpha）
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif ch == 3:
            pass
        else:
            # 非法通道数，强转 3 通道
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Invalid image ndim: {img.ndim}")

    # 再次尺寸校验，防止裁剪为 0
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Empty image after crop: shape={img.shape}")

    # dtype 统一到 uint8（PaddleOCR 预处理会自己做归一化）
    if img.dtype != np.uint8:
        # 常见错误：float32 0~1 或 0~255；统一到 0~255 uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img