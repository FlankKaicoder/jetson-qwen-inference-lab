"""Record whether the frozen Phase 2 venv can export the requested ONNX models."""
import importlib.util
import json
from pathlib import Path


def main() -> None:
    result = {
        "onnx_module_available": importlib.util.find_spec("onnx") is not None,
        "requested_opset": 17,
        "requested_shapes": ["1x1024", "32x1024"],
        "requested_linear_dimensions": ["1024->1024", "1024->2048", "1024->3072"],
    }
    if not result["onnx_module_available"]:
        result["status"] = "BLOCKED_NO_ONNX_PACKAGE"
        result["reason"] = "Phase 2.1 forbids installing Python packages in the frozen venv."
    else:
        result["status"] = "AVAILABLE_NOT_EXECUTED"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
