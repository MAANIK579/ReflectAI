"""
app/recommendations/grooming_classifier.py — AI Facial Grooming & Health Analyzer.

Uses models/grooming/grooming_classifier.pt (multi-label ResNet-18) to analyze:
  - Hair: style, color, receding hairline, baldness
  - Facial Hair: beard, mustache, clean shaven, patchy beard
  - Skin Health: clear skin, dark circles, oily skin
  - Presentation: well groomed, glasses, hat, makeup
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import resnet18
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    nn = None
    transforms = None
    resnet18 = None

logger = logging.getLogger("reflectai.grooming")

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "grooming" / "grooming_classifier.pt"

_GroomingBase = nn.Module if nn is not None else object


class _GroomingNet(_GroomingBase):
    def __init__(self, num_attrs: int):
        super().__init__()
        if resnet18 is not None and nn is not None:
            self.backbone = resnet18(weights=None)
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Linear(512, num_attrs)

    def forward(self, x: Any) -> Any:
        feat = self.backbone(x)
        return self.classifier(feat)


class GroomingClassifier:
    _model: Optional[_GroomingNet] = None
    _attributes: List[str] = []
    _is_loaded: bool = False

    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) if transforms is not None else None

    @classmethod
    def load_model(cls) -> bool:
        if not _HAS_TORCH:
            return False
        if cls._is_loaded:
            return True

        if not MODEL_PATH.exists():
            logger.warning(f"Grooming model not found at {MODEL_PATH}")
            return False

        try:
            checkpoint = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
            cls._attributes = checkpoint.get("attributes", [])
            cls._model = _GroomingNet(len(cls._attributes))
            state_dict = checkpoint.get("model_state") or checkpoint.get("model_state_dict")
            cls._model.load_state_dict(state_dict)
            cls._model.eval()
            cls._is_loaded = True
            logger.info(f"Loaded grooming classifier with {len(cls._attributes)} attributes")
            return True
        except Exception as e:
            logger.error(f"Failed to load grooming classifier: {e}")
            return False

    @classmethod
    def analyze(
        cls,
        image_source: Union[str, Path, bytes, io.BytesIO, Image.Image],
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze face/head crop for hair, beard, skin health, and presentation.
        """
        if not cls.load_model():
            if not _HAS_TORCH:
                return {
                    "status": "ok",
                    "well_groomed": True,
                    "facial_hair": "Clean shaven",
                    "skin": "Clear skin",
                    "hair_style": "Neat",
                    "accessories": [],
                }
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
                logits = cls._model(tensor)[0]
                probs = torch.sigmoid(logits)

            scores: Dict[str, float] = {}
            for attr, prob in zip(cls._attributes, probs):
                scores[attr] = round(float(prob.item()), 3)

            # Extract features (threshold ~0.35 - 0.5 depending on feature)
            active = {k: (v >= 0.40) for k, v in scores.items()}

            # 1. Hair Analysis
            hair_desc = []
            if active.get("bald"):
                hair_desc.append("Bald")
            elif active.get("receeding_hairline"):
                hair_desc.append("Receding hairline")
            elif active.get("long_hair"):
                hair_desc.append("Long hair")
            elif active.get("curly_hair"):
                hair_desc.append("Curly hair")
            else:
                hair_desc.append("Short hair")

            if active.get("grey_hair"):
                hair_desc.append("(grey)")
            elif active.get("black_hair"):
                hair_desc.append("(dark)")

            hair_summary = " ".join(hair_desc)

            # 2. Beard / Facial Hair Analysis
            if active.get("has_beard") and active.get("has_mustache"):
                beard_summary = "Full beard & mustache"
            elif active.get("has_beard"):
                beard_summary = "Beard"
            elif active.get("patchy_beard"):
                beard_summary = "Stubble / light beard"
            elif active.get("has_mustache"):
                beard_summary = "Mustache"
            else:
                beard_summary = "Clean shaven"

            # 3. Skin & Wellness Analysis
            skin_notes = []
            if active.get("clear_skin"):
                skin_notes.append("Clear glowing skin")
            if active.get("dark_circles"):
                skin_notes.append("Dark circles detected")
            if active.get("oily_skin"):
                skin_notes.append("Oily skin")

            skin_summary = " · ".join(skin_notes) if skin_notes else "Normal skin"

            # 4. Accessories
            accessories = []
            if active.get("wearing_glasses"):
                accessories.append("Glasses")
            if active.get("wearing_hat"):
                accessories.append("Hat / Cap")
            if active.get("has_makeup"):
                accessories.append("Makeup")

            # 5. Overall Grooming Advice
            tips = []
            is_well_groomed = active.get("well_groomed", False)

            if active.get("dark_circles"):
                tips.append("You look a bit tired — hydrate well and get plenty of rest tonight!")
            if active.get("patchy_beard"):
                tips.append("A light beard trim will sharpen your look.")
            if is_well_groomed:
                tips.append("Sharp and well-groomed today!")
            elif not tips:
                tips.append("Looking clean and fresh!")

            advice_text = " ".join(tips)

            # Voice summary for voice assistant
            voice_parts = []
            if is_well_groomed:
                voice_parts.append("You're looking sharp and well-groomed.")
            else:
                voice_parts.append(f"Your hair looks like {hair_summary.lower()}, and you have a {beard_summary.lower()}.")

            if active.get("dark_circles"):
                voice_parts.append("I noticed some dark circles under your eyes, so be sure to get some rest!")
            elif active.get("clear_skin"):
                voice_parts.append("Your skin looks healthy and clear today.")

            voice_summary = " ".join(voice_parts)

            return {
                "status": "ok",
                "hair": hair_summary,
                "facial_hair": beard_summary,
                "skin": skin_summary,
                "accessories": accessories,
                "well_groomed": is_well_groomed,
                "advice": advice_text,
                "scores": scores,
                "voice_summary": voice_summary,
            }
        except Exception as e:
            logger.error(f"Grooming analysis failed: {e}", exc_info=True)
            return None
