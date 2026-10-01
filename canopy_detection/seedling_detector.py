"""High-resolution seedling candidate generation and classification.

This module is deliberately independent from the Detectree2 crown detector.
The mature-canopy model produces the baseline mask first; this module may add
small, separately-scored candidates without changing that baseline mask.

The runtime checkpoint is optional.  When it is not present, callers can use
the existing legacy color supplement as a safe compatibility fallback.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn


@dataclass
class SeedlingDetectorConfig:
    """Runtime configuration for the isolated seedling branch."""

    mode: str = "hybrid"
    model_path: Optional[str] = None
    classifier_threshold: float = 0.90
    isolated_classifier_threshold: float = 0.97
    input_size: int = 64
    crop_size_m: float = 0.75
    min_area_m2: float = 0.0008
    max_area_m2: float = 0.025
    max_bbox_m: float = 0.45
    min_saturation: float = 28.0
    min_value: float = 40.0
    min_exg: float = 10.0
    neighbor_radius_m: float = 1.25
    min_neighbors: int = 1
    duplicate_radius_m: float = 0.10
    marker_radius_m: float = 0.08
    local_contrast_radius_m: float = 0.45
    min_local_value_gain: float = 10.0
    min_local_exg_gain: float = 18.0
    min_chroma: float = 0.012
    micro_supplement: bool = True
    micro_min_area_m2: float = 0.00005
    micro_max_area_m2: float = 0.0008
    micro_max_bbox_m: float = 0.22
    micro_min_saturation: float = 48.0
    micro_min_value: float = 55.0
    micro_min_exg: float = 16.0
    micro_min_chroma: float = 0.009
    micro_min_local_value_gain: float = 6.0
    micro_min_local_exg_gain: float = 12.0
    micro_min_neighbors: int = 1
    micro_support_radius_m: float = 0.06
    micro_min_support_pixels: int = 2
    batch_size: int = 64

    @classmethod
    def from_mapping(cls, values: Optional[Dict[str, Any]] = None) -> "SeedlingDetectorConfig":
        """Build a config from the detector's runtime-tuning dictionary."""
        values = values or {}
        aliases = {
            "seedling_detection_mode": "mode",
            "seedling_classifier_path": "model_path",
            "seedling_classifier_threshold": "classifier_threshold",
            "seedling_isolated_classifier_threshold": "isolated_classifier_threshold",
            "seedling_crop_size_m": "crop_size_m",
            "seedling_input_size": "input_size",
            "seedling_batch_size": "batch_size",
            "seedling_min_area_m2": "min_area_m2",
            "seedling_max_area_m2": "max_area_m2",
            "seedling_max_bbox_m": "max_bbox_m",
            "seedling_min_saturation": "min_saturation",
            "seedling_min_value": "min_value",
            "seedling_min_exg": "min_exg",
            "seedling_neighbor_radius_m": "neighbor_radius_m",
            "seedling_min_neighbors": "min_neighbors",
            "seedling_duplicate_radius_m": "duplicate_radius_m",
            "seedling_marker_radius_m": "marker_radius_m",
            "seedling_local_contrast_radius_m": "local_contrast_radius_m",
            "seedling_min_local_value_gain": "min_local_value_gain",
            "seedling_min_local_exg_gain": "min_local_exg_gain",
            "seedling_min_chroma": "min_chroma",
            "seedling_micro_supplement": "micro_supplement",
            "seedling_micro_min_area_m2": "micro_min_area_m2",
            "seedling_micro_max_area_m2": "micro_max_area_m2",
            "seedling_micro_max_bbox_m": "micro_max_bbox_m",
            "seedling_micro_min_saturation": "micro_min_saturation",
            "seedling_micro_min_value": "micro_min_value",
            "seedling_micro_min_exg": "micro_min_exg",
            "seedling_micro_min_chroma": "micro_min_chroma",
            "seedling_micro_min_local_value_gain": "micro_min_local_value_gain",
            "seedling_micro_min_local_exg_gain": "micro_min_local_exg_gain",
            "seedling_micro_min_neighbors": "micro_min_neighbors",
            "seedling_micro_support_radius_m": "micro_support_radius_m",
            "seedling_micro_min_support_pixels": "micro_min_support_pixels",
        }
        accepted = set(cls.__dataclass_fields__.keys())
        kwargs = {}
        for key, value in values.items():
            mapped_key = aliases.get(key, key)
            if mapped_key in accepted:
                kwargs[mapped_key] = value
        config = cls(**kwargs)
        config.mode = str(config.mode or "hybrid").strip().lower()
        if config.mode not in {"legacy", "hybrid", "off", "disabled"}:
            config.mode = "hybrid"
        config.classifier_threshold = float(np.clip(config.classifier_threshold, 0.5, 0.999))
        config.isolated_classifier_threshold = float(
            np.clip(config.isolated_classifier_threshold, config.classifier_threshold, 0.999)
        )
        config.input_size = max(32, int(config.input_size))
        config.batch_size = max(1, int(config.batch_size))
        config.min_neighbors = max(0, int(config.min_neighbors))
        config.micro_min_neighbors = max(0, int(config.micro_min_neighbors))
        config.micro_min_support_pixels = max(1, int(config.micro_min_support_pixels))
        config.micro_support_radius_m = max(0.03, float(config.micro_support_radius_m))
        return config


@dataclass
class SeedlingCandidate:
    """A color/shape candidate before or after classifier scoring."""

    x: float
    y: float
    area_px: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    mean_saturation: float
    mean_value: float
    mean_exg: float
    mean_chroma: float
    value_gain: float
    exg_gain: float
    neighbor_count: int = 0
    classifier_score: Optional[float] = None
    accepted: bool = False
    rejection_reason: Optional[str] = None
    is_micro: bool = False
    strong_color: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TinySeedlingClassifier(nn.Module):
    """Small CPU-friendly RGB patch classifier used by the hybrid branch."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.15),
            nn.Linear(96, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _default_model_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    configured = os.getenv("MANGROVISION_SEEDLING_CLASSIFIER_PATH")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else root / path
    return root / "models" / "seedling_classifier" / "best.pt"


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load either a TorchScript module or a state-dict checkpoint."""
    try:
        scripted = torch.jit.load(str(path), map_location=device)
        scripted.eval()
        return scripted, {"format": "torchscript"}
    except Exception:
        pass

    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(path), map_location=device)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model, {"format": "module"}

    if not isinstance(checkpoint, dict):
        raise ValueError("Seedling checkpoint must be a TorchScript module, module, or dictionary")

    model = TinySeedlingClassifier().to(device)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Seedling checkpoint does not contain a state_dict")
    cleaned_state_dict = {
        str(key).removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata["format"] = "state_dict"
    return model, metadata


class SeedlingDetector:
    """Generate, classify, and render isolated small-seedling candidates."""

    def __init__(self, config: Optional[SeedlingDetectorConfig] = None, device: str = "cpu") -> None:
        self.config = config or SeedlingDetectorConfig()
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.model_metadata: Dict[str, Any] = {}
        self.model_path: Optional[str] = None
        self.load_error: Optional[str] = None
        if self.config.mode == "hybrid":
            self._load_model()

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def status(self) -> str:
        if self.config.mode in {"off", "disabled"}:
            return "disabled"
        if self.config.mode == "legacy":
            return "legacy"
        return "hybrid" if self.available else "fallback_legacy"

    def _load_model(self) -> None:
        path = Path(self.config.model_path) if self.config.model_path else _default_model_path()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        self.model_path = str(path)
        if not path.exists():
            self.load_error = f"checkpoint not found: {path}"
            return
        try:
            self.model, self.model_metadata = _load_checkpoint(path, self.device)
            calibrated_threshold = self.model_metadata.get("classifier_threshold")
            if calibrated_threshold is not None and self.config.classifier_threshold == 0.90:
                self.config.classifier_threshold = float(np.clip(float(calibrated_threshold), 0.5, 0.999))
                self.config.isolated_classifier_threshold = max(
                    self.config.classifier_threshold,
                    self.config.isolated_classifier_threshold,
                )
        except Exception as exc:
            self.model = None
            self.load_error = f"checkpoint could not be loaded: {exc}"

    @staticmethod
    def _metric_kernel_size(distance_m: float, gsd: float, fallback_px: int = 9) -> int:
        if distance_m <= 0:
            return 0
        size = int(round(distance_m / gsd)) if gsd > 0 else fallback_px
        size = max(3, min(size, 151))
        if size % 2 == 0:
            size += 1
        return size

    @staticmethod
    def _color_masks(image: np.ndarray, config: SeedlingDetectorConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        bgr = image.astype(np.float32)
        blue = bgr[:, :, 0]
        green = bgr[:, :, 1]
        red = bgr[:, :, 2]
        excess_green = (2.0 * green) - red - blue
        chroma = (green - np.maximum(red, blue)) / np.maximum(red + green + blue, 1.0)

        strong = (
            (hue >= 16) & (hue <= 65)
            & (saturation >= max(config.min_saturation, 35.0))
            & (value >= config.min_value)
            & (excess_green >= max(config.min_exg, 10.0))
            & (green > (red + 4.0))
            & (green > (blue + 2.0))
            & (chroma >= config.min_chroma)
        )
        pale = (
            (hue >= 16) & (hue <= 72)
            & (saturation >= config.min_saturation)
            & (value >= max(config.min_value, 58.0))
            & (excess_green >= config.min_exg)
            & (green > (red + 2.0))
            & (green > (blue + 1.0))
            & (chroma >= config.min_chroma)
        )
        return (strong | pale).astype(np.uint8), saturation, value, excess_green, chroma

    def generate_candidates(
        self,
        image: np.ndarray,
        base_mask: Optional[np.ndarray],
        gsd: float,
    ) -> Tuple[List[SeedlingCandidate], Dict[str, Any]]:
        """Create compact color/shape candidates before learned scoring."""
        if image is None or image.ndim != 3 or gsd <= 0:
            return [], {"candidate_generation_status": "invalid_input"}

        config = self.config
        pixel_mask, saturation, value, excess_green, chroma = self._color_masks(image, config)
        contrast_kernel = self._metric_kernel_size(config.local_contrast_radius_m, gsd, 47)
        local_value = cv2.blur(value.astype(np.float32), (contrast_kernel, contrast_kernel))
        local_exg = cv2.blur(excess_green.astype(np.float32), (contrast_kernel, contrast_kernel))
        micro_support_kernel = self._metric_kernel_size(config.micro_support_radius_m, gsd, 9)
        micro_support = cv2.boxFilter(
            pixel_mask.astype(np.float32),
            -1,
            (micro_support_kernel, micro_support_kernel),
            normalize=False,
            borderType=cv2.BORDER_REPLICATE,
        )
        micro_local_value = cv2.blur(
            value.astype(np.float32),
            (micro_support_kernel, micro_support_kernel),
        )
        micro_local_exg = cv2.blur(
            excess_green.astype(np.float32),
            (micro_support_kernel, micro_support_kernel),
        )

        duplicate_mask = np.zeros(pixel_mask.shape, dtype=np.uint8)
        if isinstance(base_mask, np.ndarray) and base_mask.shape == pixel_mask.shape:
            duplicate_mask = (base_mask > 0).astype(np.uint8)
            duplicate_kernel_size = self._metric_kernel_size(config.duplicate_radius_m, gsd, 9)
            duplicate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (duplicate_kernel_size, duplicate_kernel_size),
            )
            duplicate_mask = cv2.dilate(duplicate_mask, duplicate_kernel, iterations=1)

        normal_min_area_px = max(1, int(round(config.min_area_m2 / (gsd ** 2))))
        micro_min_area_px = max(1, int(round(config.micro_min_area_m2 / (gsd ** 2))))
        micro_max_area_px = max(
            micro_min_area_px,
            int(round(config.micro_max_area_m2 / (gsd ** 2))),
        )
        min_area_px = micro_min_area_px if config.micro_supplement else normal_min_area_px
        max_area_px = max(min_area_px, int(round(config.max_area_m2 / (gsd ** 2))))
        max_bbox_px = max(3, int(round(config.max_bbox_m / gsd)))
        micro_max_bbox_px = max(3, int(round(config.micro_max_bbox_m / gsd)))

        labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(pixel_mask, 8)
        candidates: List[SeedlingCandidate] = []
        rejected_shape = 0
        rejected_existing = 0
        rejected_color = 0
        micro_candidate_count = 0
        micro_rejected_shape = 0
        micro_rejected_color = 0
        micro_single_candidate_count = 0
        micro_single_rejected_color = 0
        micro_single_rejected_support = 0
        image_h, image_w = image.shape[:2]

        for label_idx in range(1, labels_count):
            x, y, width, height, area = stats[label_idx].tolist()
            if area < min_area_px or area > max_area_px or width > max_bbox_px or height > max_bbox_px:
                rejected_shape += 1
                continue
            is_micro = bool(
                config.micro_supplement
                and area >= micro_min_area_px
                and area <= micro_max_area_px
            )
            if is_micro:
                micro_candidate_count += 1
                if width > micro_max_bbox_px or height > micro_max_bbox_px:
                    rejected_shape += 1
                    micro_rejected_shape += 1
                    micro_candidate_count -= 1
                    continue
            aspect = max(width, height) / max(1, min(width, height))
            if aspect > (4.0 if is_micro else 6.0):
                rejected_shape += 1
                if is_micro:
                    micro_rejected_shape += 1
                    micro_candidate_count -= 1
                continue

            component = labels[y : y + height, x : x + width] == label_idx
            if np.count_nonzero(component & (duplicate_mask[y : y + height, x : x + width] > 0)) > 0:
                rejected_existing += 1
                continue

            component_sat = saturation[y : y + height, x : x + width][component]
            component_val = value[y : y + height, x : x + width][component]
            component_exg = excess_green[y : y + height, x : x + width][component]
            component_chroma = chroma[y : y + height, x : x + width][component]
            mean_sat = float(np.mean(component_sat))
            mean_val = float(np.mean(component_val))
            mean_exg = float(np.mean(component_exg))
            mean_chroma = float(np.mean(component_chroma))
            cx, cy = centroids[label_idx]
            center_x = int(np.clip(round(cx), 0, image_w - 1))
            center_y = int(np.clip(round(cy), 0, image_h - 1))
            value_gain = mean_val - float(local_value[center_y, center_x])
            exg_gain = mean_exg - float(local_exg[center_y, center_x])
            micro_value_gain = mean_val - float(micro_local_value[center_y, center_x])
            micro_exg_gain = mean_exg - float(micro_local_exg[center_y, center_x])

            strong_color = (
                mean_sat >= 55.0
                and mean_val >= 45.0
                and mean_exg >= 22.0
                and mean_chroma >= max(config.min_chroma, 0.012)
                and (exg_gain >= config.min_local_exg_gain or value_gain >= config.min_local_value_gain)
            )
            pale_contrast = (
                mean_sat >= config.min_saturation
                and mean_val >= 70.0
                and mean_exg >= 12.0
                and mean_chroma >= config.min_chroma
                and value_gain >= config.min_local_value_gain
                and exg_gain >= config.min_local_exg_gain
            )
            micro_color = (
                mean_sat >= config.micro_min_saturation
                and mean_val >= config.micro_min_value
                and mean_exg >= config.micro_min_exg
                and mean_chroma >= config.micro_min_chroma
                and max(value_gain, micro_value_gain) >= config.micro_min_local_value_gain
                and max(exg_gain, micro_exg_gain) >= config.micro_min_local_exg_gain
            )
            dark_mud_like = (
                mean_val < (float(local_value[center_y, center_x]) - 4.0)
                and exg_gain < (config.min_local_exg_gain + 2.0)
                and mean_sat < 75.0
            )
            if is_micro and not micro_color:
                rejected_color += 1
                micro_rejected_color += 1
                continue
            if is_micro and area == 1:
                micro_single_candidate_count += 1
                support_pixels = int(round(float(micro_support[center_y, center_x])))
                if support_pixels < config.micro_min_support_pixels:
                    rejected_color += 1
                    micro_rejected_color += 1
                    micro_single_rejected_support += 1
                    continue
            if not is_micro and (dark_mud_like or not (strong_color or pale_contrast)):
                rejected_color += 1
                continue

            candidates.append(
                SeedlingCandidate(
                    x=float(cx),
                    y=float(cy),
                    area_px=int(area),
                    bbox_x=int(x),
                    bbox_y=int(y),
                    bbox_w=int(width),
                    bbox_h=int(height),
                    mean_saturation=mean_sat,
                    mean_value=mean_val,
                    mean_exg=mean_exg,
                    mean_chroma=mean_chroma,
                    value_gain=float(value_gain),
                    exg_gain=float(exg_gain),
                    is_micro=is_micro,
                    strong_color=bool(strong_color),
                )
            )

        radius_px = max(1.0, config.neighbor_radius_m / gsd)
        radius_sq = radius_px * radius_px
        for index, candidate in enumerate(candidates):
            neighbors = 0
            for other_index, other in enumerate(candidates):
                if index == other_index:
                    continue
                dx = candidate.x - other.x
                dy = candidate.y - other.y
                if (dx * dx) + (dy * dy) <= radius_sq:
                    neighbors += 1
            candidate.neighbor_count = neighbors

        metadata = {
            "seedling_candidate_count": int(len(candidates)),
            "seedling_rejected_existing": int(rejected_existing),
            "seedling_rejected_shape": int(rejected_shape),
            "seedling_rejected_color": int(rejected_color),
            "seedling_rejected_neighbor": 0,
            "seedling_micro_candidate_count": int(micro_candidate_count),
            "seedling_micro_rejected_shape": int(micro_rejected_shape),
            "seedling_micro_rejected_color": int(micro_rejected_color),
            "seedling_micro_single_candidate_count": int(micro_single_candidate_count),
            "seedling_micro_single_rejected_color": int(micro_single_rejected_color),
            "seedling_micro_single_rejected_support": int(micro_single_rejected_support),
            "seedling_micro_min_area_m2": float(config.micro_min_area_m2),
            "seedling_neighbor_radius_m": float(config.neighbor_radius_m),
            "seedling_min_neighbors": int(config.min_neighbors),
        }
        return candidates, metadata

    def _crop_tensor(self, image: np.ndarray, candidate: SeedlingCandidate, gsd: float) -> torch.Tensor:
        side = max(self.config.input_size, int(round(self.config.crop_size_m / gsd)))
        half = max(1, side // 2)
        center_x = int(round(candidate.x))
        center_y = int(round(candidate.y))
        padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REFLECT_101)
        x = center_x + half
        y = center_y + half
        crop = padded[y - half : y + half, x - half : x + half]
        crop = cv2.resize(crop, (self.config.input_size, self.config.input_size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - 0.5) / 0.5
        return torch.from_numpy(np.transpose(crop, (2, 0, 1)))

    @torch.inference_mode()
    def _score_candidates(
        self,
        image: np.ndarray,
        candidates: Sequence[SeedlingCandidate],
        gsd: float,
    ) -> List[float]:
        if self.model is None:
            return []
        scores: List[float] = []
        for start in range(0, len(candidates), self.config.batch_size):
            batch_candidates = candidates[start : start + self.config.batch_size]
            batch = torch.stack([self._crop_tensor(image, candidate, gsd) for candidate in batch_candidates])
            logits = self.model(batch.to(self.device))
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            scores.extend(float(value) for value in probabilities.detach().cpu().tolist())
        return scores

    def detect(
        self,
        image: np.ndarray,
        base_mask: Optional[np.ndarray],
        gsd: float,
    ) -> Tuple[np.ndarray, List[SeedlingCandidate], Dict[str, Any]]:
        """Return an isolated marker mask, accepted candidates, and telemetry."""
        empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if self.config.mode in {"off", "disabled"}:
            return empty_mask, [], {"seedling_detection_mode": "disabled", "seedling_classifier_available": False}
        if self.config.mode == "legacy":
            return empty_mask, [], {"seedling_detection_mode": "legacy", "seedling_classifier_available": False}
        if self.model is None:
            return empty_mask, [], {
                "seedling_detection_mode": "fallback_legacy",
                "seedling_classifier_available": False,
                "seedling_classifier_path": self.model_path,
                "seedling_classifier_error": self.load_error,
            }

        candidates, metadata = self.generate_candidates(image, base_mask, gsd)
        if not candidates:
            metadata.update({
                "seedling_detection_mode": "hybrid",
                "seedling_classifier_available": True,
                "seedling_classifier_threshold": float(self.config.classifier_threshold),
                "seedling_supplement_count": 0,
                "seedling_supplement_pixels": 0,
                "seedling_micro_supplement_count": 0,
            })
            return empty_mask, [], metadata

        scores = self._score_candidates(image, candidates, gsd)
        rejected_classifier = 0
        rejected_neighbor = 0
        rejected_duplicate = 0
        accepted: List[SeedlingCandidate] = []
        for candidate, score in zip(candidates, scores):
            candidate.classifier_score = float(score)
            if score < self.config.classifier_threshold:
                candidate.rejection_reason = "classifier"
                rejected_classifier += 1
                continue
            required_neighbors = (
                self.config.micro_min_neighbors
                if candidate.is_micro
                else self.config.min_neighbors
            )
            if candidate.is_micro and candidate.area_px == 1:
                required_neighbors = 0
            if not candidate.is_micro and candidate.strong_color:
                required_neighbors = min(required_neighbors, 1)
            if (
                candidate.neighbor_count < required_neighbors
                and score < self.config.isolated_classifier_threshold
            ):
                candidate.rejection_reason = "neighbor"
                rejected_neighbor += 1
                continue
            accepted.append(candidate)

        accepted.sort(key=lambda item: float(item.classifier_score or 0.0), reverse=True)
        min_distance_px = max(0.0, self.config.duplicate_radius_m / gsd)
        min_distance_sq = min_distance_px * min_distance_px
        deduplicated: List[SeedlingCandidate] = []
        for candidate in accepted:
            too_close = any(
                ((candidate.x - other.x) ** 2) + ((candidate.y - other.y) ** 2) <= min_distance_sq
                for other in deduplicated
            )
            if too_close:
                candidate.accepted = False
                candidate.rejection_reason = "duplicate"
                rejected_duplicate += 1
                continue
            candidate.accepted = True
            deduplicated.append(candidate)

        marker_radius_px = max(2, int(round(self.config.marker_radius_m / gsd)))
        mask = np.zeros_like(empty_mask)
        for candidate in deduplicated:
            cv2.circle(mask, (int(round(candidate.x)), int(round(candidate.y))), marker_radius_px, 255, -1)

        metadata.update({
            "seedling_detection_mode": "hybrid",
            "seedling_classifier_available": True,
            "seedling_classifier_path": self.model_path,
            "seedling_classifier_threshold": float(self.config.classifier_threshold),
            "seedling_isolated_classifier_threshold": float(self.config.isolated_classifier_threshold),
            "seedling_supplement_count": int(len(deduplicated)),
            "seedling_supplement_pixels": int(np.count_nonzero(mask)),
            "seedling_micro_supplement_count": int(
                sum(1 for candidate in deduplicated if candidate.is_micro)
            ),
            "seedling_rejected_classifier": int(rejected_classifier),
            "seedling_rejected_neighbor": int(rejected_neighbor),
            "seedling_rejected_duplicate": int(rejected_duplicate),
            "seedling_candidates": [candidate.to_dict() for candidate in candidates],
            "seedling_accepted_candidates": [candidate.to_dict() for candidate in deduplicated],
        })
        return mask, deduplicated, metadata


def checkpoint_metadata(path: Optional[str] = None) -> Dict[str, Any]:
    """Return checkpoint metadata without loading a model into the process."""
    model_path = Path(path) if path else _default_model_path()
    if not model_path.exists():
        return {"path": str(model_path), "exists": False}
    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
        metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
        return {"path": str(model_path), "exists": True, "metadata": metadata}
    except Exception as exc:
        return {"path": str(model_path), "exists": True, "error": str(exc)}


__all__ = [
    "SeedlingCandidate",
    "SeedlingDetector",
    "SeedlingDetectorConfig",
    "TinySeedlingClassifier",
    "checkpoint_metadata",
]
