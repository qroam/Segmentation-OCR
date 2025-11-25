# A Segmentation-OCR Two Stage Pipeline

## Motivation

<!-- ![Image failed to load](/img/pipeline.jpg "Two stage pipeline") -->

Current **optical character recognition (OCR)** technology has achieved near-perfect performance in many industrial application scenarios. However, in some situations, large amounts of noise exist in addition to the textual information we care about, and the target text is often submerged within complex environmental information. This poses challenges for directly applying OCR models.

For example, in photos taken with a camera, the text to be recognized may occupy only a very small area and may appear at a tilted angle. The target text may also be mixed with other text (that we don't care about) in the environment. In scanned images of specific types of documents, our requirement may be to extract only particular information—such as a document ID—which often has unique visual features. An OCR model will treat all text equally, making post-processing difficult.

Such cases necessitate combining **instance segmentation** with **OCR** to create a two-stage pipeline system. A customized instance segmentation model trained on domain-specific data can precisely locate and crop the required text instances from the input image, which are then passed to the OCR model for recognition. This greatly improves performance on these scenarios and also facilitates structured extraction of document content based on visual features.

## Quick Start
```python
cd ..
python interface.py
```

## Dependencies Installation

### 1. Create conda env
```bash
conda create -n yolo_paddle_env python=3.9

conda activate yolo_paddle_env
```

### 2. YOLO dependencies
- YOLO is based on **PyTorch**
- Refer to https://pytorch.org/get-started/previous-versions/ to get PyTorch version corresponding to your CUDA

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 3. paddleOCR dependencies
- paddleOCR is based on **paddlepaddle**
- Refer to https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html  to get paddlepaddle version corresponding to your CUDA

```bash
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

python -m pip install "paddleocr[all]"
```
## Notice

Caution for the folder name containing your paddle models, since it may loaded successfully when particular characters occur in path.

## Limitation

`PyTorch` and `PaddlePaddle` are two different deep learning frameworks. Although it is possible to install them in the same environment and run simultaneously as shown in this example, this requires using specific compatible versions. In practice, loading `PyTorch` and `PaddlePaddle` at the same time often leads to incompatibility issues, making debugging troublesome. Therefore, we emphasize installing the latest versions as this can help reduce such problems to some extent. Nevertheless, using two different frameworks simultaneously still presents certain challenges and is not a recommended practice. This leaves a big room for future optimization of this project.