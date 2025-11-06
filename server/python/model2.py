import torch
import io
import sys
import logging
from torch import nn
from torchvision import models
import json
import os

SPLIT_LAYER = 1


class EfficientNetPart2(nn.Module):
    def __init__(self, full_model: models.EfficientNet, split_layer: int):
        super().__init__()
        modules = list(full_model.features)
        self.part = nn.Sequential(*modules[split_layer:])
        self.avgpool = full_model.avgpool
        self.classifier = full_model.classifier

    def forward(self, x: torch.Tensor):
        x = self.part(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_labels():
    candidates = [
        "python/imagenet_classes.json",
        "imagenet_classes.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    return {i: name for i, name in enumerate(loaded)}
                if isinstance(loaded, dict):
                    return {int(k): v for k, v in loaded.items()}
            except Exception:
                continue
    return {i: f"Class {i}" for i in range(1000)}


def main():
    intermediate_file = sys.argv[1] if len(sys.argv) > 1 else "intermediate_features.bin"
    result_file = sys.argv[2] if len(sys.argv) > 2 else "final_result.bin"
    log_file = sys.argv[3] if len(sys.argv) > 3 else "../logs/server_logs/session_model2.log"

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    logging.info("model2 started. intermediate=%s result=%s", intermediate_file, result_file)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        full_model = models.efficientnet_b0(pretrained=True).to(device).eval()
        part2 = EfficientNetPart2(full_model, SPLIT_LAYER).to(device).eval()

        with open(intermediate_file, "rb") as f:
            buffer = io.BytesIO(f.read())

        data = torch.load(buffer)
        intermediate = data["intermediate"].to(device)
        image_paths = data.get("image_paths", [])
        batch_size = data.get("batch_size", intermediate.shape[0] if hasattr(intermediate, "shape") else 1)

        logging.info("Received intermediate shape=%s batch_size=%s", list(intermediate.shape), batch_size)

        first_conv = None
        for m in part2.part.modules():
            if isinstance(m, torch.nn.Conv2d):
                first_conv = m
                break
        if first_conv is not None:
            expected = first_conv.in_channels
            actual = intermediate.shape[1]
            if expected != actual:
                msg = f"Channel mismatch: server expects {expected}, got {actual}"
                logging.error(msg)
                raise RuntimeError(msg)

        with torch.no_grad():
            logits = part2(intermediate)
            probabilities = torch.nn.functional.softmax(logits, dim=1)

        labels_map = load_labels()

        batch_results = []
        for i, (img_path, probs) in enumerate(zip(image_paths, probabilities)):
            top5 = probs.topk(5)
            top_indices = top5.indices.cpu().tolist()
            top_probs = top5.values.cpu().tolist()

            predictions = []
            for idx, p in zip(top_indices, top_probs):
                predictions.append({
                    "class_index": idx,
                    "class_name": labels_map.get(idx, f"Class {idx}"),
                    "probability": p,
                })

            top_idx = top_indices[0] if top_indices else None
            top_name = labels_map.get(top_idx, f"Class {top_idx}") if top_idx is not None else None

            batch_results.append({
                "image_path": img_path,
                "top_class": top_idx,
                "top_class_name": top_name,
                "top_probability": top_probs[0] if top_probs else 0.0,
                "predictions": predictions,
            })

        buffer = io.BytesIO()
        torch.save({
            "logits": logits.cpu(),
            "probabilities": probabilities.cpu(),
            "batch_results": batch_results,
            "batch_size": batch_size,
            "image_paths": image_paths
        }, buffer)

        result_data = buffer.getvalue()
        with open(result_file, "wb") as f:
            f.write(result_data)
        logging.info("Saved %d bytes to %s", len(result_data), result_file)

    except Exception as e:
        logging.error("Exception in model2: %s", e)
        try:
            with open(result_file, "wb") as f:
                torch.save({"error": str(e)}, f)
        except Exception:
            logging.error("Failed to write error result file")
        sys.exit(1)


if __name__ == "__main__":
    main()