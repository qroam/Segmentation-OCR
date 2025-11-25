#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
# from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import random

from collections import Counter
import os
import yaml

from seg_ocr import SegOCRPipeline
pipeline = SegOCRPipeline()

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


def draw_instance_segmentation_from_yolo(image_path, all_masks, class_names=None, display=True, output_path=None):
    # 读取图片
    # img = cv2.imread(image_path)
    # 读取图片（支持中文路径）
    
    if class_names is None:
        class_names = []
    
    if type(image_path) == str:
        img = imread_unicode(image_path)
    elif type(image_path) == np.ndarray:
        # img = Image.fromarray(image_path)
        # img = cv2.imdecode(image_path)
        img = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
        
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


# =========================
# 处理函数（示例占位）
# 传入：RGB 的 numpy 数组（H,W,3）
# 返回：processed_img（PIL.Image 或 RGB numpy），text（str）
# =========================
def process_frame(frame_rgb: np.ndarray):
    # 这里演示：在右上角画一个小矩形，并返回一句文本
    
    all_masks, all_cropped, all_extracted_texts_for_each_obj = pipeline(frame_rgb)
    # output_data = pipeline(frame_rgb)
    # text = f'当前帧识别集装箱号：{" | ".join(all_extracted_texts_for_each_obj)}'
    return draw_instance_segmentation_from_yolo(frame_rgb, all_masks, class_names=None,), all_extracted_texts_for_each_obj# display=True, output_path=None)
    


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Seg-ORC demo")
        self.geometry("1100x700")

        # ====== 顶部：按钮区 ======
        topbar = ttk.Frame(self)
        topbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        self.btn_upload = ttk.Button(topbar, text="Upload a video", command=self.on_upload)
        self.btn_upload.pack(side=tk.LEFT)

        # ====== 中部：两侧图像区（左原右处理）======
        # mid = ttk.Frame(self)
        # mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # left_frame = ttk.LabelFrame(mid, text="原始帧（每秒一帧）")
        # right_frame = ttk.LabelFrame(mid, text="处理结果")
        # left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        # right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # # 左右两个显示面板
        # self.left_panel = tk.Label(left_frame, bg="#111", fg="#ccc", anchor="center")
        # self.left_panel.pack(fill=tk.BOTH, expand=True)
        # self.right_panel = tk.Label(right_frame, bg="#111", fg="#ccc", anchor="center")
        # self.right_panel.pack(fill=tk.BOTH, expand=True)
        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 关键：两列等权，保证 50/50
        mid.columnconfigure(0, weight=1, uniform="cols")
        mid.columnconfigure(1, weight=1, uniform="cols")
        mid.rowconfigure(0, weight=1)

        left_frame = ttk.LabelFrame(mid, text="Original Frame (1FPS)")
        right_frame = ttk.LabelFrame(mid, text="Instance Segmentation")

        # 用 grid 而不是 pack
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        # 关键：不要让子控件改变 frame 尺寸
        left_frame.grid_propagate(False)
        right_frame.grid_propagate(False)

        self.left_panel  = tk.Label(left_frame,  bg="#111", fg="#ccc", anchor="center")
        self.right_panel = tk.Label(right_frame, bg="#111", fg="#ccc", anchor="center")

        # 让 label 充满各自的 frame
        self.left_panel.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.right_panel.place(relx=0, rely=0, relwidth=1, relheight=1)


        # ====== 底部：文本输出 ======
        bottom = ttk.LabelFrame(self, text="Output Texts")
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)
        self.text_output = tk.Text(bottom, height=4, wrap="word", font=("Arial", 14))
        self.text_output.pack(fill=tk.X, padx=8, pady=8)

        # ====== 播放控制相关状态 ======
        self.cap = None
        self.delay_ms = 10               # 解码推进的小延迟（不等于 1s）
        self.fps = 25
        self.frame_interval = 25         # 多少帧抽 1 帧（≈ fps）
        self._frame_index = 0
        self._next_show_index = 0
        self._timer_id = None

        self.total_frames = 0
        self._finished = False

        self.all_text = []

        # 用于持有 PhotoImage 的引用，防止被 GC
        self._left_photo = None
        self._right_photo = None

        # 关闭时清理
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ========== 事件：上传视频 ==========
    def on_upload(self):
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.ts"), ("all files", "*.*")]
        )
        if not path:
            return

        # 若已有播放，先停止释放
        self.stop_video()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Can not open the video")
            return

        self.cap = cap

        # 取 FPS（未知时给默认 25）
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-3:
            fps = 25.0
        self.fps = fps
        self.frame_interval = max(1, int(round(self.fps)))  # 每秒抽 1 帧
        self._frame_index = 0
        self._next_show_index = 0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # 可能为0，容错靠read失败
        self.frame_interval = max(1, int(round(self.fps)))
        self._frame_index = 0
        self._next_show_index = 0
        self._finished = False

        # 文本输出一行提示
        self.write_text(f"Video has been loaded successfully: {path}\nFPS≈{self.fps:.2f}, processing at FPS={self.frame_interval} \n")

        # 启动定时解码循环
        self.schedule_next()

    # ========== 循环调度 ==========
    def schedule_next(self):
        # 用 after 驱动，不阻塞 UI
        self._timer_id = self.after(self.delay_ms, self._pump_one_step)

        # self.all_text.extend(text)
        

    # ========== 解码推进 + 每秒抽一帧处理 ==========
    def _pump_one_step(self):
        if self.cap is None:
            return
        
        # 若已到最后一秒对应的抽样阈值，也可直接结束（当 total_frames 有效时）
        if self.total_frames > 0 and self._next_show_index >= self.total_frames:
            self.finish_video()
            return

        ok, frame_bgr = self.cap.read()
        if not ok:
            # 真正读到末尾（或解码失败）=> 结束并弹窗
            self.finish_video()
            return
            # 到末尾：回到第一帧继续（也可选择停止）
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_index = 0
                self._next_show_index = 0
                self.schedule_next()
            else:
                self.stop_video()
            return

        # 记录当前帧序号
        cur_idx = self._frame_index
        self._frame_index += 1

        # 是否该“每秒抽样显示”
        if cur_idx == self._next_show_index:
            self._next_show_index += self.frame_interval

            # --- 左侧：显示原始帧（BGR→RGB→PIL→PhotoImage） ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            left_img = self.to_tk_image(frame_rgb, fit_widget=self.left_panel)
            self.left_panel.configure(image=left_img)
            self._left_photo = left_img  # 保持引用

            # --- 处理函数：传入 RGB numpy，返回处理图 + 文本 ---
            processed_img, text = process_frame(frame_rgb)
            # self.all_text.extend(text)
            for t in text:
                if not t in self.all_text:
                    self.all_text.append(t)
            current_text_to_display = f'Texts in current frame: {" | ".join(text)}'

            # 兼容返回类型（PIL.Image 或 RGB numpy）
            proc_pil = self.ensure_pil_rgb(processed_img)
            right_img = self.to_tk_image(proc_pil, fit_widget=self.right_panel)
            self.right_panel.configure(image=right_img)
            self._right_photo = right_img

            # 底部文本更新（覆盖 or 追加，这里用覆盖再追加一空行）
            self.text_output.delete("1.0", tk.END)
            # self.text_output.insert(tk.END, current_text_to_display + "\n")
            # self.all_text = list(set(self.all_text))
            all_text_to_display = f'All Texts in this video recognized so far ({len(self.all_text)}): {" | ".join(self.all_text)}'
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, current_text_to_display + "\n" + all_text_to_display + "\n")

        # 继续安排下一次
        self.schedule_next()

    # ========== 工具：将 RGB numpy 或 PIL 转为 Tk 显示 ==========
    def to_tk_image(self, img_rgb, fit_widget: tk.Label):
        """img_rgb: 可以是 RGB 的 numpy 或 PIL.Image"""
        pil = self.ensure_pil_rgb(img_rgb)
        # 根据面板当前大小做等比缩放（防止过大）
        w = max(1, fit_widget.winfo_width())
        h = max(1, fit_widget.winfo_height())
        # 初次启动时可能为 1x1，可给一个默认最大宽度
        if w <= 1 or h <= 1:
            w, h = 520, 320
        pil = self.thumbnail_keep_ratio(pil, (w, h))
        return ImageTk.PhotoImage(pil)

    @staticmethod
    def ensure_pil_rgb(img):
        """把输入统一转为 PIL.Image (RGB)"""
        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                return img.convert("RGB")
            return img
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                # 灰度转 RGB
                return Image.fromarray(img).convert("RGB")
            # 假定是 RGB（如果你的处理函数返回的是 BGR，请自行改成 cv2.cvtColor(img, cv2.COLOR_BGR2RGB)）
            return Image.fromarray(img.astype(np.uint8))
        else:
            raise TypeError("processed image must be `PIL.Image` or `numpy.ndarray`")

    @staticmethod
    def thumbnail_keep_ratio(pil_img: Image.Image, max_size):
        pil = pil_img.copy()
        pil.thumbnail(max_size, Image.Resampling.LANCZOS)
        return pil

    # ========== 文本输出助手 ==========
    def write_text(self, s: str):
        self.text_output.insert(tk.END, s)
        self.text_output.see(tk.END)

    # ========== 停止/清理 ==========
    def stop_video(self):
        if self._timer_id is not None:
            try:
                self.after_cancel(self._timer_id)
            except Exception:
                pass
            self._timer_id = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def on_close(self):
        self.stop_video()
        self.destroy()
    

    def finish_video(self):
        if self._finished:
            return
        self._finished = True
        self.stop_video()  # 取消 after + 释放 cap
        messagebox.showinfo("Finished", "Current video has been finished")

if __name__ == "__main__":
    App().mainloop()
