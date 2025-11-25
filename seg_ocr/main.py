#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
# import shlex
import os
from pathlib import Path
import time
from datetime import datetime
from typing import Union, Optional, List, Dict
from tqdm import tqdm
import re

import numpy as np
import cv2
from PIL import Image

from ultralytics import YOLO

from paddleocr import PaddleOCR

from .configs import *
from .utils import pil_to_numpy, numpy_to_pil, sanitize_for_ocr
from .check import validate_full


def load_ocr_pipeline(device="gpu", *args, **kwargs):
    # 这个device必须写gpu或gpu:0, 不要写成cuda:0, paddle不认
    ocr_pipeline = PaddleOCR(
        # text_recognition_model_name="PP-OCRv5_server_rec",
        use_doc_orientation_classify=False, # Use use_doc_orientation_classify to enable/disable document orientation classification model
        use_doc_unwarping=False, # Use use_doc_unwarping to enable/disable document unwarping module
        use_textline_orientation=True, # Use use_textline_orientation to enable/disable textline orientation classification model
        # device="cpu",
        device=device,
        # device="gpu:0", # Use device to specify GPU for model inference
        # doc_orientation_classify_model_dir
        # doc_unwarping_model_dir
        text_detection_model_dir=TEXT_DETECTION_MODEL_PATH,
        textline_orientation_model_dir=TEXT_ORIENTATION_MODEL_PATH,
        text_recognition_model_dir=TEXT_RECOGNITION_MODEL_PATH,
    )
    return ocr_pipeline


class SegOCRPipeline:
    def __init__(self, **kwargs):
        self._load_segmentation_model(model_path=SEGMENTATION_MODEL_PATH, **kwargs)
        self._load_ocr_pipeline(**kwargs)

    def _load_segmentation_model(self, model_path:str, **kwargs):
        model = YOLO(model_path)
        self.seg_model = model
        self.seg_model.to("cuda:0")
        print(f"Segmentation Mpdel has been loaded:{model_path}, device = {self.seg_model.device}")

    def _load_ocr_pipeline(self, **kwargs):
        self.ocr_pipeline = load_ocr_pipeline(**kwargs)
    
    def _seg_inference(self, input_root_dir:str, conf:float, iou:float,) -> list[dict]:
        seg_results = self.seg_model.predict(
            source=input_root_dir,
            save=False,  # 保存带检测框的图像/视频
            save_txt=False,  # 保存检测框的文本文件（YOLO格式）
            save_conf=False,  # 在文本文件中保存置信度
            # project=args.input_root_dir,  # 结果保存到指定目录
            # name=args.name,  # 在输出目录下创建子文件夹名称
            conf=conf,   # 默认值
            iou=iou     # 默认值
        )

        seg_result = seg_results[0]
        
        boxes = seg_result.boxes  # Boxes 对象
        xyxy = boxes.xyxy.cpu().tolist()  # tensor [N,4]，每行是 [x1, y1, x2, y2]
        conf = boxes.conf.cpu().tolist()  # tensor [N,]
        class_ids = boxes.cls.cpu().tolist()   # tensor [N,]

        masks = seg_result.masks
        if masks:
            mask_xyn = masks.xyn#.cpu().tolist()
        else:
            mask_xyn = []
            assert class_ids == []

        all_masks = []
        for i, (coords, class_id, _xyxy, _conf) in enumerate(zip(mask_xyn, class_ids, xyxy, conf)):
            mask_data = {
                "class_id": class_id,
                "coords": coords,
                "xyxy": _xyxy,
                "conf": _conf,
            }
            all_masks.append(mask_data)
            print(f"======== Segmentation output: [{i+1}]/{len(mask_xyn)} ========")
            print(mask_data)
            
        return all_masks
    

    def _slice_mask(self, img: np.ndarray, all_masks: list[dict], target_classes: list[str]=None, ) -> List[np.ndarray]:
        # pil_img = Image.open(img_path).convert("RGB")
        # img = pil_to_numpy(pil_img)
        h, w = img.shape[:2]
        
        all_cropped = []
        for idx, mask_data in enumerate(all_masks):
            class_id = mask_data["class_id"]
            if type(target_classes) == list:
                if class_id not in target_classes:
                    continue
            # ========== 归一化坐标 → 像素坐标 ==========
            coords = mask_data["coords"]
            coords[:, 0] = coords[:, 0] * w
            coords[:, 1] = coords[:, 1] * h
            polygon = coords.astype(np.int32)

            # ========== 生成 mask ==========
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)

            # ========== 抠图 ==========
            cropped = cv2.bitwise_and(img, img, mask=mask)
            cropped = np.dstack((cropped, np.expand_dims(mask, axis=-1)), )

            # ========== 根据 polygon 的外接矩形裁剪（可选） ==========
            x, y, ww, hh = cv2.boundingRect(polygon)
            cropped = cropped[y:y+hh, x:x+ww]
            all_cropped.append(cropped)
        
        return all_cropped

    def _ocr_inference(self, img: np.ndarray,) -> Dict:
        ocr_results = self.ocr_pipeline.predict(sanitize_for_ocr(img))
        
        all_extracted_texts = ""
        for i, result in enumerate(ocr_results):
            
            all_extracted_texts += " ".join(result["rec_texts"])
            all_extracted_texts += " "  # TODO
            
        return all_extracted_texts
    
    # @staticmethod
    def _check(self, code11: str) -> tuple[bool, str]:
        try:
            is_correct, expected, given, normalized_str = validate_full(code11)
            return is_correct, normalized_str
        except Exception as e:
            return False, None

    def __call__(self, img_path: Union[str, np.ndarray], conf=0.5, iou=0.1, post_processing=False) -> list:
        if type(img_path) == str:
            pil_img = Image.open(img_path).convert("RGB")
            img = pil_to_numpy(pil_img)
        elif type(img_path) == np.ndarray:
            img = img_path
        else:
            raise TypeError(f"type(img_path) = {type(img_path)}")

        all_masks = self._seg_inference(img, conf, iou,)
        all_cropped = self._slice_mask(img, all_masks,)
        all_extracted_texts = []
        for cropped in all_cropped:
            extracted_texts_single_crop = self._ocr_inference(cropped)
            all_extracted_texts.append(extracted_texts_single_crop)
        
        print("OCR output:", all_extracted_texts)

        
        all_extracted_texts_after_check = []
        if post_processing:
            output_data = []
            assert len(all_extracted_texts) == len(all_masks)
            for extracted_texts_single_crop, mask_data in zip(all_extracted_texts, all_masks):
                is_correct, normalized_str = self._check(extracted_texts_single_crop)
                if is_correct:
                    all_extracted_texts_after_check.append(normalized_str)
                    output_data.append([
                        mask_data["xyxy"][0], mask_data["xyxy"][1], mask_data["xyxy"][2], mask_data["xyxy"][3], mask_data["conf"], normalized_str,
                    ])
            print("OCR output after post processing:", all_extracted_texts_after_check)
        else:
            all_extracted_texts_after_check = all_extracted_texts
        
        return all_masks, all_cropped, all_extracted_texts_after_check
        
    

if __name__ == "__main__":
    # conda activate yolo_paddle_env-win64
    # cd ..
    # python main.py -i "xxx.jpg"

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", "-i", type=str, required=True)
    args = parser.parse_args()

    pipeline = SegOCRPipeline()
    pipeline(args.img_path)