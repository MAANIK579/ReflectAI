"""
app/recommendations/clothing_classifier.py — AI Fashion Recognition Service.

Uses the trained multi-task ResNet-18 model (models/clothing/fashion_classifier.pt)
to automatically classify clothing photos into:
  - Category: top, bottom, footwear, outerwear, accessory
  - Color: base color (e.g., White, Black, Navy Blue)
  - SubCategory: fine-grained category (e.g., Topwear, Bottomwear, Shoes)
  - Suggested Name: automatic name proposal
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

logger = logging.getLogger("reflectai.classifier")

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "clothing" / "fashion_classifier.pt"

# Map subCategory to ReflectAI core categories ('top', 'bottom', 'footwear', 'outerwear', 'accessory')
SUB_CATEGORY_MAP = {
    "topwear": "top",
    "dress": "top",
    "saree": "top",
    "apparel set": "top",
    "innerwear": "top",
    "loungewear and nightwear": "top",
    "bottomwear": "bottom",
    "shoes": "footwear",
    "flip flops": "footwear",
    "sandal": "footwear",
    "shoe accessories": "footwear",
    "socks": "footwear",
    "mufflers": "outerwear",
    "scarves": "outerwear",
    "gloves": "outerwear",
    "jackets": "outerwear",
    "stoles": "outerwear",
    "bags": "accessory",
    "belts": "accessory",
    "cufflinks": "accessory",
    "eyewear": "accessory",
    "jewellery": "accessory",
    "ties": "accessory",
    "umbrellas": "accessory",
    "wallets": "accessory",
    "watches": "accessory",
    "wristbands": "accessory",
    "headwear": "accessory",
    "accessories": "accessory",
}

MASTER_CATEGORY_MAP = {
    "apparel": "top",
    "footwear": "footwear",
    "accessories": "accessory",
}


class _MultiTaskNet(nn.Module):
    def __init__(self, targets: Dict[str, int]):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict({k: nn.Linear(512, v) for k, v in targets.items()})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {k: head(features) for k, head in self.heads.items()}


class ClothingClassifier:
    _model: Optional[_MultiTaskNet] = None
    _encoder_classes: Dict[str, list] = {}
    _is_loaded: bool = False

    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @classmethod
    def load_model(cls) -> bool:
        """Load model weights and label encoders once into memory."""
        if cls._is_loaded:
            return True

        if not MODEL_PATH.exists():
            logger.warning(f"Fashion classifier model not found at {MODEL_PATH}")
            return False

        try:
            logger.info(f"Loading fashion classifier model from {MODEL_PATH}...")
            checkpoint = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)

            num_classes = checkpoint.get("num_classes_per_target", {})
            cls._encoder_classes = checkpoint.get("encoder_classes", {})

            model = _MultiTaskNet(num_classes)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()

            cls._model = model
            cls._is_loaded = True
            logger.info("Fashion classifier model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load fashion classifier model: {e}", exc_info=True)
            return False

    @classmethod
    def classify(
        cls,
        image_source: Union[str, Path, bytes, io.BytesIO, Image.Image],
        region: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Classify a clothing photo into category, subCategory, color, and gender.
        Supports region hint ('top' or 'bottom') and aspect ratio prior calibration.
        """
        if not cls.load_model():
            return None

        try:
            if isinstance(image_source, (str, Path)):
                img = Image.open(str(image_source)).convert("RGB")
            elif isinstance(image_source, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_source)).convert("RGB")
            elif isinstance(image_source, io.BytesIO):
                img = Image.open(image_source).convert("RGB")
            elif isinstance(image_source, Image.Image):
                img = image_source.convert("RGB")
            else:
                logger.error(f"Unsupported image input type: {type(image_source)}")
                return None

            tensor = cls._transform(img).unsqueeze(0)

            with torch.no_grad():
                preds = cls._model(tensor)

            raw_results = {}
            for target, logits in preds.items():
                logits_copy = logits[0].clone()
                classes = cls._encoder_classes.get(target, [])

                # Prior calibration for Bottomwear:
                # The training dataset has a 6x class imbalance favoring Topwear over Bottomwear.
                # If the image is vertical (pants/jeans) or explicitly cropped from lower body:
                if target == "subCategory" and "Bottomwear" in classes:
                    b_idx = classes.index("Bottomwear")
                    ar = img.height / max(1, img.width)
                    if region == "bottom":
                        logits_copy[b_idx] += 0.85
                    elif ar > 1.2:
                        logits_copy[b_idx] += 0.45
                    elif region == "top":
                        logits_copy[b_idx] -= 0.35

                idx = int(logits_copy.argmax().item())
                if idx < len(classes):
                    raw_results[target] = classes[idx]
                else:
                    raw_results[target] = "Unknown"

            sub_cat = raw_results.get("subCategory", "")
            master_cat = raw_results.get("masterCategory", "")
            color = raw_results.get("baseColour", "")
            gender = raw_results.get("gender", "")

            # Map to core ReflectAI category
            sub_key = sub_cat.lower().strip()
            master_key = master_cat.lower().strip()

            core_category = SUB_CATEGORY_MAP.get(
                sub_key,
                MASTER_CATEGORY_MAP.get(master_key, "top" if region != "bottom" else "bottom")
            )
            if region == "bottom" and core_category != "bottom":
                core_category = "bottom"

            # Build human-friendly suggested name
            name_parts = []
            if color and color.lower() != "unknown":
                name_parts.append(color)
            if sub_cat and sub_cat.lower() != "unknown":
                name_parts.append(sub_cat)
            else:
                name_parts.append(core_category.title())

            suggested_name = " ".join(name_parts)

            return {
                "category": core_category,
                "color": color,
                "sub_category": sub_cat,
                "master_category": master_cat,
                "gender": gender,
                "suggested_name": suggested_name,
            }
        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            return None
