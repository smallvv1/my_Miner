"""Small runtime compatibility shim for the PPU PyTorch environment."""

import os


if os.environ.get("MINER_PPU_RUNTIME") == "1":
    import torch

    # The installed torchvision wheel expects these schemas during import,
    # while the PPU build intentionally does not ship torchvision C++ ops.
    _torchvision_library = torch.library.Library("torchvision", "DEF")
    for _schema in (
        "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor",
        "qnms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor",
    ):
        try:
            _torchvision_library.define(_schema)
        except RuntimeError:
            pass
