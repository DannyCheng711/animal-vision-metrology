"""Check package imports and compiled operators without data or model downloads.

Run from the project root: python -m run.check_environment
"""
import platform
import sys
from importlib import import_module
from importlib.metadata import version


def main():
    # ====================================================
    # Step 1: Check the supported platform and Python version
    # ====================================================
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The locked environment requires Linux x86_64")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The locked environment requires Python 3.11")
    print("Python:", platform.python_version(), flush=True)

    # ====================================================
    # Step 2: Check dependencies and project entry points
    # ====================================================
    for package, module in (
        ("numpy", "numpy"), ("torch", "torch"), ("torchvision", "torchvision"),
        ("mmcv", "mmcv"), ("mmengine", "mmengine"), ("mmdet", "mmdet"),
        ("mmpose", "mmpose.apis"), ("opencv-python", "cv2"),
        ("ultralytics", "ultralytics"), ("segment-anything", "segment_anything"),
        ("pycocotools", "pycocotools.mask"),
    ):
        import_module(module)
        print(f"{package}: {version(package)}", flush=True)
    for module in (
        "run.inference", "run.animal_segmentation_evaluation",
        "run.eye_localization_evaluation", "run.eye_segmentation_evaluation",
        "run.eye_distance_evaluation",
    ):
        import_module(module)

    # ====================================================
    # Step 3: Check MMCV on CPU and torchvision on available devices
    # ====================================================
    import torch
    from torchvision.ops import nms as torchvision_nms
    from mmcv.ops import nms as mmcv_nms

    print("PyTorch CUDA runtime:", torch.version.cuda, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    cpu_boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]])
    cpu_scores = torch.tensor([0.9, 0.8])
    _, indices = mmcv_nms(cpu_boxes, cpu_scores, 0.5)
    if indices.tolist() != [0]:
        raise RuntimeError("MMCV CPU NMS check failed")
    print("MMCV compiled operator OK: cpu", flush=True)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    for device in devices:
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
        scores = torch.tensor([0.9, 0.8], device=device)
        if torchvision_nms(boxes, scores, 0.5).tolist() != [0]:
            raise RuntimeError(f"torchvision NMS check failed on {device}")
        print(f"torchvision compiled operator OK: {device}", flush=True)
    print("Environment check passed; model loading and inference require separate validation.")


if __name__ == "__main__":
    main()
