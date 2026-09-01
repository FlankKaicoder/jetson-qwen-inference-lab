"""Phase 2.0 synthetic probe entry point; full probes require a passing TorchAO import gate."""
import json
import torch

result = {
    "torch": torch.__version__,
    "torch_file": torch.__file__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    "int4pack_op": hasattr(torch.ops.aten, "_weight_int4pack_mm"),
}
try:
    import torchao  # noqa: F401
    result["torchao_import"] = "PASS"
except Exception as exc:
    result["torchao_import"] = "FAIL"
    result["torchao_error"] = repr(exc)
print(json.dumps(result, default=str, sort_keys=True))
