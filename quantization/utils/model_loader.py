"""
InternVL-U 模型加载器 — 用于量化搜索流程。
加载 InternVLUPipeline 的 VLM 部分，供逐层 CKA 搜索使用。
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def load_internvlu(
    model_path: str,
    gpu_ids: Optional[str] = None,
    torch_dtype=torch.bfloat16,
) -> Dict[str, Any]:
    """
    加载 InternVL-U 的 VLM（InternVLUChatModel）和 processor。

    Returns:
        {
            'model': InternVLUChatModel,
            'tokenizer': tokenizer,
            'processor': InternVLUProcessor,
            'new_token_ids': None,  # 兼容 Bagel 接口
        }
    """
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    model_path = Path(model_path).resolve()
    if (model_path / "model_index.json").exists():
        model_root = model_path
    elif (model_path / "model" / "model_index.json").exists():
        model_root = model_path / "model"
    else:
        model_root = model_path

    # internvlu 包所在的仓库根目录
    internvlu_repo = Path(__file__).resolve().parent.parent.parent
    if str(internvlu_repo) not in sys.path:
        sys.path.insert(0, str(internvlu_repo))

    from internvlu import InternVLUPipeline

    print(f"Loading InternVL-U from {model_root} ...")
    pipeline = InternVLUPipeline.from_pretrained(
        str(model_root),
        torch_dtype=torch_dtype,
    )
    pipeline.to("cuda")

    vlm = pipeline.vlm.eval()
    processor = pipeline.processor
    tokenizer = processor.tokenizer

    if getattr(vlm, "img_context_token_id", None) is None:
        img_ctx = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        vlm.img_context_token_id = img_ctx

    return {
        "model": vlm,
        "tokenizer": tokenizer,
        "processor": processor,
        "new_token_ids": None,
    }
