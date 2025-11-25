from pathlib import Path
import os

# 以本文件为基准，定位到项目根目录 seg_ocr/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 允许通过环境变量覆盖（可选，用于部署/测试切换）
MODEL_ROOT = Path(os.getenv("SEG_OCR_MODEL_ROOT", PROJECT_ROOT))

# 模型路径（绝对路径）
SEGMENTATION_MODEL_PATH   = MODEL_ROOT / "seg_models" / "<your-segmentation-model-name>"
TEXT_DETECTION_MODEL_PATH = MODEL_ROOT / "ocr_models" / "paddle" / "PP-OCRv5_server_det"
TEXT_ORIENTATION_MODEL_PATH = MODEL_ROOT / "ocr_models" / "paddle" / "PP-LCNet_x1_0_textline_ori"
TEXT_RECOGNITION_MODEL_PATH = MODEL_ROOT / "ocr_models" / "paddle" / "PP-OCRv5_server_rec"

SEGMENTATION_MODEL_PATH   = str(SEGMENTATION_MODEL_PATH)
TEXT_DETECTION_MODEL_PATH = str(TEXT_DETECTION_MODEL_PATH)
TEXT_ORIENTATION_MODEL_PATH = str(TEXT_ORIENTATION_MODEL_PATH)
TEXT_RECOGNITION_MODEL_PATH = str(TEXT_RECOGNITION_MODEL_PATH)
