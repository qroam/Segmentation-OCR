# 实例分割-光学字符识别（OCR）两阶段系统

**English Version README:** [README.md](https://github.com/qroam/Segmentation-OCR/blob/main/README.md)

## 动机

![图片未加载](/img/demo.jpg "Two stage pipeline")

当今光学字符识别（OCR）技术在行业应用场景中已经实现了近乎完美的水平。然而，在一些场景中，除了我们所关注的文本信息之外还存在大量的噪声，关注的文本信息被淹没在环境信息中，这对直接使用OCR模型带来了挑战。例如，在拍摄的照片中，待识别的文本可能所占面积很小，且角度为倾斜的；待识别的文本可能与环境中其他文本混杂在一起，OCR模型会将它们一视同仁地识别出来，这给后处理带来了困难；在特定类型文档的扫描件中，可能需求只是抽取特定的信息，例如文档编号，而它们往往具备特定的视觉特征。此时，非常有必要将实例分割和光学字符识别结合起来，形成一个两阶段系统。由在特定数据上定制的实例分割模型从输入图片中精确地框选出所需要识别的文字实例，再交给OCR模型识别，能够很大程度地提升这类任务上的性能，这样也有利于实现文档内容基于视觉特征的结构化抽取。

## 快速测试
```python
cd ..
python interface.py
```

## 依赖安装

### 1. 创建conda环境
```bash
conda create -n yolo_paddle_env python=3.9

conda activate yolo_paddle_env
```

### 2. 安装yolo相关依赖
- YOLO依赖于PyTorch
- 参考https://pytorch.org/get-started/previous-versions/ 获取CUDA对应版本的pytorch

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 3. 安装paddleOCR相关依赖
- paddleOCR依赖于paddlepaddle
- 参考https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html 获取CUDA对应版本的paddlepaddle

```bash
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

python -m pip install "paddleocr[all]"
```
## 注意

不要将本项目尤其是paddle模型文件放在含有中文字符的路径下，paddle模型加载遇到这样的路径会出错！

## 局限性与展望

PyTorch和paddlepaddle是两套不同的深度学习框架，虽然它们确实可以像本示例中这样，安装在同一个环境中并同时运行，但这需要安装特定的版本才能实现。事实上，在运行时同时加载PyTorch和paddlepaddle在很多情况下都会遇到不兼容的情况，这给调试带来了很大麻烦。因此，我们强调尽量安装最新版（即上述列出的版本），能一定程度避免此类问题。同时，这种同时使用两套不同框架的做法仍存在一定问题，这留给未来很大的优化空间。