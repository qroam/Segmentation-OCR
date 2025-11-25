#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path

import cv2
import os
import yaml
import random

from collections import Counter

def draw_yolo_boxes(image_path, label_path, output_path=None, class_names=None):
    
    if os.path.exists(class_names):
        with open(class_names, "r", encoding="utf-8") as f:
            # class_names = yaml.safe_load(f.read())["names"]
            class_names = yaml.safe_load(f.read()).get("names", "?")
        
    
    # 读取图像
    # image = cv2.imread(image_path)
    # img = Image.open(image_path)
    # img_array = np.array(img)
    # image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # print(image)
    h, w = image.shape[:2]

    # 读取标签文件
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) not in [5, 6]:
            continue  # 跳过格式错误的行

        class_id = int(parts[0])
        x_center, y_center, box_w, box_h = map(float, parts[1:5])
        confidence = float(parts[5]) if len(parts) == 6 else None

        # 转换为左上角、右下角坐标
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # 准备标签文字
        print(class_names)
        if class_names:
            label = class_names[class_id] if class_id < len(class_names) else f"id:{class_id}"
        else:
            label = str(class_id)
        if confidence is not None:
            label += f" {confidence:.2f}"

        # 绘制标签背景框
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 6), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 保存图像
    # image.show()
    # plt.imshow(image)
    # plt.show()
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Save image to {output_path}")
    return image




def imread_unicode(path):
    """支持中文路径的图像读取"""
    # 用 numpy.fromfile + cv2.imdecode 读取中文路径的图片
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def imwrite_unicode(path, img):
    """支持中文路径的图像保存"""
    # 用 cv2.imencode + tofile 保存带中文路径的图片
    ext = '.' + path.split('.')[-1]
    success, buf = cv2.imencode(ext, img)
    if success:
        buf.tofile(path)


def load_yolo_segmentation(txt_path):
    """
    读取 YOLO 实例分割格式的标注文件
    每一行格式: class_id x1 y1 x2 y2 ... (归一化坐标，点数一般>=6)
    """
    objects = []
    # with open(txt_path, 'r') as f:
    with open(txt_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            objects.append((cls_id, coords))
    return objects


def resize_for_display(img, max_size=800):
    """
    缩放图片以便显示，最长边不超过 max_size
    """
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)  # 只缩小，不放大
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def draw_instance_segmentation(image_path, txt_path, class_names=None, display=True, output_path=None):
    # 读取图片
    # img = cv2.imread(image_path)
    # 读取图片（支持中文路径）
    
    if class_names is None:
        class_names = []
        
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"Failed to read the image: {image_path}")
    h, w = img.shape[:2]
    # h, w = img.shape[:2]

    # 读取YOLO格式结果
    objects = load_yolo_segmentation(txt_path)

    overlay = img.copy()

    for cls_id, coords in objects:
        # YOLO 分割坐标是归一化的 (x,y)，需转为像素坐标
        pts = np.array([[int(coords[i] * w), int(coords[i+1] * h)] for i in range(0, len(coords), 2)])
        
        # 随机生成一个颜色
        color = [random.randint(0, 255) for _ in range(3)]
        
        # 绘制填充的透明多边形
        cv2.fillPoly(overlay, [pts], color)

        # 在多边形边界绘制线条
        cv2.polylines(img, [pts], True, color, 2)

        # 在多边形第一个点附近写上类别标签
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        cv2.putText(img, label, (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # 将透明层叠加到原图
    alpha = 0.4  # 透明度
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img

    # if output_path:
    #     # 保存结果
    #     # cv2.imwrite(output_path, img)
    #     # 保存结果（支持中文路径）
    #     imwrite_unicode(output_path, img)
    #     print(f"结果已保存到 {output_path}")
    
    # # if display:
    # #     # 是否显示结果
    # #     cv2.imshow("Instance Segmentation Result", img)
    # #     cv2.waitKey(0)   # 等待任意键关闭
    # #     cv2.destroyAllWindows()
    # # 是否显示结果
    # if display:
    #     img_show = resize_for_display(img, max_size=800)
    #     cv2.imshow("Instance Segmentation Result", img_show)
    #     cv2.waitKey(0)   # 等待任意键关闭
    #     cv2.destroyAllWindows()


def draw_instance_segmentation_from_yolo(image_path, all_masks, class_names=None, display=True, output_path=None):
    # 读取图片
    # img = cv2.imread(image_path)
    # 读取图片（支持中文路径）
    
    if class_names is None:
        class_names = []
        
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"Failed to read the image: {image_path}")
    h, w = img.shape[:2]
    # h, w = img.shape[:2]

    # 读取YOLO格式结果
    # objects = load_yolo_segmentation(txt_path)

    overlay = img.copy()

    for mask in all_masks:
        print(mask)
        cls_id, coords = mask["class_id"], mask["coords"]
        cls_id = int(cls_id)
        coords = coords.tolist()
        # print(cls_id, coords)
        # YOLO 分割坐标是归一化的 (x,y)，需转为像素坐标
        # pts = np.array([[int(xy[0] * w), int(xy[1] * h)] for xy in coords])
        pts = np.array([[int(xy[0]), int(xy[1])] for xy in coords])

        print(pts)
        
        # 随机生成一个颜色
        color = [random.randint(0, 255) for _ in range(3)]
        
        # 绘制填充的透明多边形
        cv2.fillPoly(overlay, [pts], color)

        # 在多边形边界绘制线条
        cv2.polylines(img, [pts], True, color, 2)

        # 在多边形第一个点附近写上类别标签
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        cv2.putText(img, label, (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # 将透明层叠加到原图
    alpha = 0.4  # 透明度
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


        
class ImageProcessorApp:
    def __init__(self, master, pipeline):
        self.master = master
        # master.title("高级图片处理工具")
        master.title("Seg+OCR")

        # 配置主窗口布局
        self.configure_layout()
        
        # 初始化图片显示尺寸
        # self.display_size = (400, 400)
        self.display_size = (800, 800)

        self.pipeline = pipeline
        

    def configure_layout(self):
        """界面布局配置"""
        # 控件框架
        control_frame = tk.Frame(self.master, pady=10)
        control_frame.pack(side=tk.TOP)
        
        # 图片显示区域
        display_frame = tk.Frame(self.master)
        display_frame.pack(side=tk.TOP, pady=10)

        # 控件配置
        self.upload_btn = tk.Button(
            control_frame,
            text="Upload an image",
            command=self.process_image_upload,
            # width=15,
            # height=2
            width=15,
            height=2
        )
        self.upload_btn.pack(side=tk.LEFT, padx=20)
        
        # 图片显示面板
        self.original_panel = tk.Label(display_frame, borderwidth=2, relief="groove")
        self.original_panel.pack(side=tk.LEFT, padx=20)
        
        self.processed_panel = tk.Label(display_frame, borderwidth=2, relief="groove")
        self.processed_panel.pack(side=tk.RIGHT, padx=20)

        self.output_panel = tk.Label(root, text="processing...", font=("微软雅黑", 16))
        self.output_panel.pack()

    
    def process_image_upload(self):
        """完整的图片处理流程"""
        file_path = filedialog.askopenfilename(
            filetypes=[("image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            try:
                # 原始图片处理
                pil_image = self.load_and_prepare_image(file_path)
                self.show_image(self.original_panel, pil_image, "input image")
                
                # 转换到numpy数组进行处理
                # np_image = np.array(pil_image)
                label_path = os.path.join(Path(file_path).parent.parent, "labels", f"{Path(file_path).stem}.txt")
                class_names = os.path.join(Path(file_path).parent.parent.parent, "data.yaml")
                processed_np, all_extracted_texts_for_each_obj = self.real_image_processing(image_path=file_path, label_path=label_path, class_names=class_names)
                
                # 转换回PIL格式并显示
                processed_pil = self.numpy_to_pil(processed_np)
                resized_pil = self.resize_with_aspect(processed_pil)
                self.show_image(self.processed_panel, resized_pil, "output image")

                # self.show_text(self.output_panel, all_extracted_texts_for_each_obj)
                self.output_panel.config(text=" | ".join(all_extracted_texts_for_each_obj))
            
            except Exception as e:
                self.show_error(str(e))
                raise e
    
    
    def load_and_prepare_image(self, path):
        """图片加载预处理"""
        image = Image.open(path)
        return self.resize_with_aspect(image)
    
    def resize_with_aspect(self, image):
        """智能尺寸调整"""
        original_width, original_height = image.size
        max_width, max_height = self.display_size
        
        ratio = min(max_width/original_width, max_height/original_height)
        new_size = (int(original_width*ratio), int(original_height*ratio))
        return image.resize(new_size, Image.LANCZOS)
    
    def numpy_to_pil(self, np_image):
        """将numpy数组转换为PIL图像（自动处理格式）"""
        if np_image.dtype != np.uint8:
            np_image = (np_image * 255).astype(np.uint8)

        if len(np_image.shape) == 3 and np_image.shape[2] == 3:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(np_image,)# mode=mode)
        
        # 自动检测通道数
        if len(np_image.shape) == 2:
            return Image.fromarray(np_image, mode='L')
        elif np_image.shape[2] == 3:
            return Image.fromarray(np_image, mode='RGB')
        elif np_image.shape[2] == 4:
            return Image.fromarray(np_image, mode='RGBA')
        else:
            raise ValueError(f"Your input numpy shape {np_image.shape} is incorrect")
    
    def real_image_processing(self, image_path, label_path, class_names=None):
        # """实际图像处理逻辑（示例：边缘检测）"""
        # # 此处替换为实际处理逻辑，例如：
        # # 使用OpenCV进行边缘检测
        # import cv2
        # gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        # edges = cv2.Canny(gray, 100, 200)
        # return edges[:, :, np.newaxis]  # 增加通道维度
        # 进行推理
        # return draw_yolo_boxes(image_path, label_path, class_names=class_names)
        all_masks, all_cropped, all_extracted_texts_for_each_obj = self.pipeline(image_path)
        return draw_instance_segmentation_from_yolo(image_path, all_masks, class_names=class_names,), all_extracted_texts_for_each_obj# display=True, output_path=None)
    
    
    def show_image(self, panel, image, title_text):
        """图像显示函数"""
        photo = ImageTk.PhotoImage(image)
        panel.config(image=photo, text=title_text, compound="top")
        panel.image = photo  # 保持引用
    
    def show_error(self, message):
        """错误提示"""
        error_window = tk.Toplevel(self.master)
        tk.Label(error_window, text=f"Error: {message}", fg="red").pack()


if __name__ == "__main__":
    from seg_ocr import SegOCRPipeline
    pipeline = SegOCRPipeline()

    root = tk.Tk()
    app = ImageProcessorApp(root, pipeline)
    root.mainloop()
    
    
