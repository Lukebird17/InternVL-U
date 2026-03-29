"""
InternVL-U 校准数据加载 & 前向辅助工具

支持理解 (UND) + 生成 (GEN) 混合校准，
让量化搜索同时覆盖两种推理路径的激活分布。
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from PIL import Image


def image_for_processor(sample_image) -> Optional[Any]:
    """将 calibration 里的 image 转为 processor 可用的 PIL 格式。"""
    if sample_image is None:
        return None
    if isinstance(sample_image, Image.Image):
        return sample_image
    if torch.is_tensor(sample_image):
        x = sample_image
        if x.dim() == 3:
            x = x.float().cpu()
            if x.max() <= 1.0:
                x = (x * 255).clamp(0, 255)
            x = x.permute(1, 2, 0).numpy().astype(np.uint8)
            return Image.fromarray(x)
    return None


class CalibrationDataLoader:
    """InternVL-U 校准数据加载器 — 理解 (UND) + 生成 (GEN) 混合"""

    DEFAULT_UND_PROMPTS = [
        "What is the capital of France? Answer the question using a single word or phrase.",
        "Is the sky blue? Answer the question using a single word or phrase.",
        "What color is grass? Answer the question using a single word or phrase.",
        "How many legs does a cat have? Answer the question using a single word or phrase.",
        "What is 2 + 2? Answer the question using a single word or phrase.",
        "Is water wet? Answer the question using a single word or phrase.",
        "What is the largest planet in our solar system? Answer the question using a single word or phrase.",
        "Is the Earth flat? Answer the question using a single word or phrase.",
        "What language is spoken in Japan? Answer the question using a single word or phrase.",
        "How many days are in a week? Answer the question using a single word or phrase.",
        "What is the boiling point of water in Celsius? Answer the question using a single word or phrase.",
        "Is the sun a star? Answer the question using a single word or phrase.",
        "What is the chemical symbol for gold? Answer the question using a single word or phrase.",
        "How many continents are there? Answer the question using a single word or phrase.",
        "What is the speed of light approximately? Answer the question using a single word or phrase.",
        "Is iron magnetic? Answer the question using a single word or phrase.",
        "Describe the main objects and their colors in this image.",
        "What is the tallest mountain in the world? Answer the question using a single word or phrase.",
        "Who wrote Romeo and Juliet? Answer the question using a single word or phrase.",
        "What is the chemical formula for water? Answer the question using a single word or phrase.",
        "How many planets are in the solar system? Answer the question using a single word or phrase.",
        "What is the currency of Japan? Answer the question using a single word or phrase.",
        "Is a whale a mammal? Answer the question using a single word or phrase.",
        "What is the smallest prime number? Answer the question using a single word or phrase.",
        "What continent is Brazil on? Answer the question using a single word or phrase.",
        "Is glass a solid or liquid? Answer the question using a single word or phrase.",
        "What gas do plants absorb? Answer the question using a single word or phrase.",
        "How many sides does a hexagon have? Answer the question using a single word or phrase.",
        "What is the freezing point of water in Fahrenheit? Answer the question using a single word or phrase.",
        "Which planet is known as the Red Planet? Answer the question using a single word or phrase.",
        "What is the main language spoken in Brazil? Answer the question using a single word or phrase.",
        "How many bones does an adult human have? Answer the question using a single word or phrase.",
    ]

    DEFAULT_GEN_PROMPTS = [
        "A beautiful sunset over the ocean with orange and purple clouds.",
        "A futuristic city with flying cars and neon lights at night.",
        "A cozy cabin in the snowy mountains during winter.",
        "A serene Japanese garden with cherry blossoms and a koi pond.",
        "A steampunk robot reading a book in a library.",
        "An underwater coral reef teeming with colorful tropical fish.",
        "A fantasy castle floating among clouds at golden hour.",
        "An astronaut riding a horse on the surface of Mars.",
        "A photorealistic portrait of a cat wearing a top hat and monocle.",
        "A dense enchanted forest with magical glowing mushrooms at twilight.",
        "A bustling medieval marketplace with merchants and colorful banners.",
        "A minimalist modern living room with floor-to-ceiling windows overlooking a lake.",
        "A dragon flying over a volcanic landscape at sunset.",
        "A field of lavender stretching to the horizon under a starry sky.",
        "A cyberpunk alley with holographic advertisements and rain reflections.",
        "A vintage biplane flying above a patchwork of farmland.",
        "A giant tree house in a tropical rainforest with rope bridges.",
        "A snow-covered village with warm lights glowing from every window.",
        "A golden retriever puppy playing in a field of sunflowers.",
        "A lighthouse on a rocky cliff during a dramatic thunderstorm.",
        "A traditional Chinese ink painting of misty mountains and rivers.",
        "A sleek sports car driving through a neon-lit tunnel at night.",
        "A whimsical hot air balloon festival with hundreds of colorful balloons.",
        "A detailed close-up of a mechanical pocket watch with visible gears.",
        "A peaceful zen garden with raked sand and smooth stones.",
        "A Viking longship sailing through icy fjords under the northern lights.",
        "An art deco skyscraper reflecting the golden light of dawn.",
        "A magical library with floating books and glowing crystals.",
        "A photorealistic bowl of ramen with steam rising from the broth.",
        "A tropical beach with crystal clear water and palm trees.",
        "A wolf howling at a full moon on a snowy mountain peak.",
        "A colorful coral garden viewed from below the ocean surface.",
    ]

    def __init__(
        self,
        num_und_samples: int = 16,
        num_gen_samples: int = 16,
        mme_data_root: Optional[str] = None,
    ):
        self.num_und_samples = num_und_samples
        self.num_gen_samples = num_gen_samples
        self.mme_data_root = mme_data_root

    def load_mme_samples(self) -> List[Dict]:
        if self.mme_data_root is None:
            return []
        data_root = Path(self.mme_data_root)
        if not data_root.exists():
            return []
        samples = []
        categories = [d.name for d in data_root.iterdir() if d.is_dir()]
        if not categories:
            return []
        samples_per_cat = max(1, self.num_und_samples // len(categories))
        for category in sorted(categories):
            cat_path = data_root / category
            txt_files = sorted(list(cat_path.glob("*.txt")))[:samples_per_cat]
            for txt_file in txt_files:
                if len(samples) >= self.num_und_samples:
                    break
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                img_name = txt_file.stem
                img_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    p = cat_path / f"{img_name}{ext}"
                    if p.exists():
                        img_path = p
                        break
                if img_path is None:
                    images_dir = cat_path / "images"
                    if images_dir.exists():
                        for ext in ['.png', '.jpg', '.jpeg']:
                            p = images_dir / f"{img_name}{ext}"
                            if p.exists():
                                img_path = p
                                break
                if img_path is None:
                    continue
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        samples.append({
                            'image_path': str(img_path),
                            'question': parts[0] + ' Answer the question using a single word or phrase.',
                            'category': category,
                        })
                        if len(samples) >= self.num_und_samples:
                            break
        return samples[:self.num_und_samples]

    def prepare_calibration_samples(self) -> List[Dict]:
        result = []

        # ---- UND samples ----
        mme_samples = self.load_mme_samples()
        if mme_samples:
            for s in mme_samples:
                try:
                    img = Image.open(s['image_path']).convert('RGB')
                    result.append({
                        'task_type': 'und',
                        'prompt': s['question'],
                        'image': img,
                        'generation_mode': 'text',
                    })
                except Exception:
                    pass
        if len(result) < self.num_und_samples:
            for p in self.DEFAULT_UND_PROMPTS[:self.num_und_samples - len(result)]:
                result.append({
                    'task_type': 'und',
                    'prompt': p,
                    'image': None,
                    'generation_mode': 'text',
                })

        und_count = len(result)

        # ---- GEN samples (T2I generation pathway) ----
        if self.num_gen_samples > 0:
            gen_prompts = self.DEFAULT_GEN_PROMPTS[:self.num_gen_samples]
            if len(gen_prompts) < self.num_gen_samples:
                reps = (self.num_gen_samples // len(self.DEFAULT_GEN_PROMPTS)) + 1
                gen_prompts = (self.DEFAULT_GEN_PROMPTS * reps)[:self.num_gen_samples]
            for p in gen_prompts:
                result.append({
                    'task_type': 'gen',
                    'prompt': p,
                    'image': None,
                    'generation_mode': 'image',
                })

        gen_count = len(result) - und_count
        print(f"    Calibration samples: {len(result)} total "
              f"({und_count} und + {gen_count} gen)")
        return result
