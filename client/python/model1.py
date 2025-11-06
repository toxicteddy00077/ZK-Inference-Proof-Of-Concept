import torch
import io
from torch import nn
from torchvision import models, transforms
from PIL import Image
import sys
import os


class EfficientNetPart1(nn.Module):
    def __init__(self, full_model: models.EfficientNet, split_layer: int):
        super().__init__()
        self.features = nn.Sequential(*list(full_model.features[:split_layer]))

    def forward(self, x: torch.Tensor):
        return self.features(x)


def find_image_path(path):
    if os.path.exists(path):
        return path
    basename = os.path.basename(path)
    candidates = [
        os.path.join("images", basename),
        os.path.join("../images", basename),
        os.path.join(os.path.dirname(__file__), "..", "images", basename),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c
    return None


def main():

    output_file = sys.argv[1]
    image_paths = sys.argv[2:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_model = models.efficientnet_b0(pretrained=True).to(device).eval()
    split_layer = 1
    part1 = EfficientNetPart1(full_model, split_layer).to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_tensors = []
    valid_image_paths = []

    print(f"Processing {len(image_paths)} images...")

    for idx, image_path in enumerate(image_paths):
        resolved = find_image_path(image_path)
        if resolved is None:
            print(f"[{idx+1}] NOT FOUND: {image_path}")
            continue
        try:
            image = Image.open(resolved).convert('RGB')
            input_tensor = preprocess(image)
        except Exception as e:
            print(f"[{idx+1}] ERROR loading {image_path}: {e}")
            continue
        batch_tensors.append(input_tensor)
        valid_image_paths.append(resolved)
        print(f"[{idx+1}] Loaded: {resolved}")

    if len(batch_tensors) == 0:
        print("No valid images to process")
        return

    batch_input = torch.stack(batch_tensors).to(device)
    print(f"Batch shape: {batch_input.shape}")

    with torch.no_grad():
        intermediate = part1(batch_input)

    buffer = io.BytesIO()
    torch.save({
        "intermediate": intermediate.cpu(),
        "shape": list(intermediate.shape),
        "image_paths": valid_image_paths,
        "batch_size": len(valid_image_paths),
    }, buffer)

    serialized_data = buffer.getvalue()
    with open(output_file, "wb") as f:
        f.write(serialized_data)

    print(f"Saved {len(serialized_data)} bytes to {output_file}")


if __name__ == "__main__":
    main()