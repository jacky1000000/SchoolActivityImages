# src/models/__init__.py

# 從同目錄下的 nextgpt_arch.py 檔案中導入核心模型類別
from .nextgpt_arch import NextGPTImageGenModel

# 定義當外部執行 "from models import *" 時，允許被導出的對象清單（選填，但符合最佳規範）
__all__ = ["NextGPTImageGenModel"]
