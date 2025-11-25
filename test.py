#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd


VIDEO_PATH = r"your-input-video.MP4"


def main(pipeline):
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        print(f"错误：视频不存在 {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误：无法打开视频 {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    if not fps or fps <= 0:
        print("无法获取有效 FPS")
        fps = 25.0

    if total_frames:
        duration_sec = math.floor(total_frames / fps)
    else:
        duration_sec = 10**9

    print(f"视频信息：FPS={fps:.4f}, 总帧数={total_frames}, 时长约={duration_sec} 秒")

    all_results = []
    # rotation_angles = [0, 180]

    sec = 0
    while sec <= duration_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"处理结束")
            break

        print(f"第 {sec} 秒")

        try:
            # angle_results = ocr_single_image(
            #     rotated, rotation_angle=angle,
            #     frame_sec=sec, frame_index=int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # )
            all_masks, all_cropped, all_extracted_texts_for_each_obj = pipeline(frame)
            all_extracted_texts_for_each_obj = pipeline(frame)
            all_results.extend(all_extracted_texts_for_each_obj)
            print(f"识别到 {len(all_extracted_texts_for_each_obj)} 项文本")
        except Exception as e:
            print(f"识别出错: {e}")

        sec += 1

    cap.release()

    if not all_results:
        print("未检测到箱号")
        return

    print(f"共计识别到 {len(all_results)} 项文本（包含重复）。")
    print(f"all_results: {all_results}")

    dedup = list(set(all_results))

    print(f"去重后剩余 {len(dedup)} 项文本。")
    print(f"dedup: {dedup}")


if __name__ == "__main__":
    from datetime import datetime
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    from seg_ocr import SegOCRPipeline
    pipeline = SegOCRPipeline()
    main(pipeline)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))