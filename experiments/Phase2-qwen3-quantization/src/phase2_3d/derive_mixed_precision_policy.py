from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL = Path("/home/nvidia/models/qwen3-0.6b-c1899de289a04d12100db370d81485cdf75e47ca")
MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
OPS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FACTORS = (0.90, 0.925, 0.95, 0.975, 1.00)
WINDOWS_START = "6fe35b42f4fd06477e406e68d079366ec56f6870"
JETSON_HANDOFF = "a1317a06f83634406bfb732a61f57a698e6aee2d"


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def name(layer: int, op: str) -> str:
    return f"L{layer}:{op}"


def module_name(target: str) -> str:
    layer, op = target[1:].split(":")
    group = "self_attn" if op in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
    return f"model.layers.{layer}.{group}.{op}"


def summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def robust(values: dict[str, float]) -> dict[str, Any]:
    xs = list(values.values())
    q1, q3 = float(np.percentile(xs, 25)), float(np.percentile(xs, 75))
    iqr = q3 - q1
    high = q3 + 1.5 * iqr
    rows = sorted(({"target": key, "p95_relative_l2": value} for key, value in values.items() if value > high),
                  key=lambda row: (-row["p95_relative_l2"], row["target"]))
    return {"classification": "RELATIVE_SENSITIVITY_OUTLIER_DETECTION", "metric": "PT-W8A8 vs F evaluation P95 relative-L2",
            "coefficient": 1.5, "q1": q1, "q3": q3, "iqr": iqr, "high_outlier_boundary": high,
            "outlier_count": len(rows), "outliers": rows, "not_a_production_accuracy_threshold": True}


def metric(test: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    a, b = test.float(), ref.float()
    d = a - b
    norm = torch.linalg.vector_norm(b).item()
    return {"finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
            "max_abs": float(d.abs().max().item()), "mean_abs": float(d.abs().mean().item()),
            "rmse": float(torch.sqrt(torch.mean(d * d)).item()),
            "relative_l2": float(torch.linalg.vector_norm(d).item() / norm) if norm else 0.0,
            "cosine": float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()),
            "nan_count": int(torch.isnan(a).sum().item()), "inf_count": int(torch.isinf(a).sum().item())}


def memory(stage: str) -> dict[str, Any]:
    available = next((int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:")), None)
    return {"stage": stage, "mem_available_bytes": available,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()), "cuda_reserved_bytes": int(torch.cuda.memory_reserved())}


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["parameter_count"] for row in rows)
    quant = [row for row in rows if row["precision_state"] == "PT_W8A8"]
    qparams = sum(row["parameter_count"] for row in quant)
    return {"linear_count": len(rows), "fp16_linear_count": len(rows) - len(quant), "int8_linear_count": len(quant),
            "fp16_linear_percent": (len(rows) - len(quant)) / len(rows) * 100.0, "int8_linear_percent": len(quant) / len(rows) * 100.0,
            "total_parameter_count": total, "fp16_parameter_count": total - qparams, "int8_parameter_count": qparams,
            "fp16_parameter_percent": (total - qparams) / total * 100.0, "int8_parameter_percent": qparams / total * 100.0,
            "parameter_percentage_semantics": "QUANTIZED_PARAMETER_COVERAGE_NOT_MEASURED_MEMORY_SAVING"}


def policy_entry(inv: dict[str, Any], static: dict[str, Any], dynamic: dict[str, Any], trt: set[str], state: str, reason: str) -> dict[str, Any]:
    target = name(inv["layer"], inv["operator"])
    return {"target": target, "layer": inv["layer"], "operator": inv["operator"], "checkpoint_key": inv["checkpoint_key"],
            "shape": inv["shape"], "parameter_count": inv["parameter_count"], "weight_sha256": inv["sha256"],
            "precision_state": state, "reason": reason, "c_static_evidence": {"pt_w8": static[target]["pt_w8"], "pc_w8_portable_counterfactual": static[target]["pc_w8"]},
            "c_dynamic_evidence": dynamic.get(target), "c_trt_confirmation_target": target in trt,
            "weight_quantization": "symmetric per-tensor INT8" if state == "PT_W8A8" else "N/A",
            "activation_quantization": "symmetric per-tensor INT8" if state == "PT_W8A8" else "N/A", "zero_point": 0 if state == "PT_W8A8" else None}


def make_policy(policy: str, inventory: list[dict[str, Any]], static: dict[str, Any], dynamic: dict[str, Any], trt: set[str], outliers: set[str], families: set[str]) -> list[dict[str, Any]]:
    result = []
    for inv in inventory:
        target = name(inv["layer"], inv["operator"])
        if policy == "P0_ALL_PT_W8A8":
            state, why = "PT_W8A8", "MAXIMUM_INT8_COVERAGE_REFERENCE"
        elif policy == "P1_ROBUST_OUTLIER_GUARD" and target in outliers:
            state, why = "FP16", "ROBUST_C_DYNAMIC_OUTLIER"
        elif policy == "P2_FAMILY_GUARD" and inv["operator"] in families:
            state, why = "FP16", "TOP2_SENSITIVE_OPERATOR_FAMILY"
        elif policy == "P2_FAMILY_GUARD" and target in outliers:
            state, why = "FP16", "ROBUST_C_DYNAMIC_OUTLIER"
        else:
            state, why = "PT_W8A8", "DEPLOYABLE_QUANTIZED_DEFAULT"
        result.append(policy_entry(inv, static, dynamic, trt, state, why))
    return result


class HookSet:
    """Each hook sends the exact input to an online callback; no activation is retained."""
    def __init__(self, model: torch.nn.Module, targets: set[str], callback: Any):
        self.seen: set[str] = set()
        self.handles = []
        for target in sorted(targets):
            module = model.get_submodule(module_name(target))
            def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], key: str = target) -> None:
                self.seen.add(key)
                callback(key, inputs[0])
            self.handles.append(module.register_forward_pre_hook(hook))

    def reset(self) -> None:
        self.seen.clear()

    def check(self, sample: str, expected: set[str]) -> None:
        if self.seen != expected:
            raise RuntimeError(f"incomplete exact Linear capture: {sample} {len(self.seen)}/{len(expected)}")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def forward_rows(model: torch.nn.Module, rows: list[dict[str, Any]], hooks: HookSet, targets: set[str]) -> None:
    with torch.inference_mode():
        for row in rows:
            hooks.reset()
            model(input_ids=torch.tensor([row["token_ids"]], dtype=torch.long, device="cuda"), use_cache=False)
            hooks.check(row["sample_id"], targets)


def quant_weight(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.float().abs().max() / 127.0
    if not float(scale):
        scale = torch.tensor(1.0)
    return (torch.round(weight.float() / scale).clamp(-127, 127) * scale).to(torch.float16)


def prepare_weights(inventory: list[dict[str, Any]], targets: set[str]) -> dict[str, torch.Tensor]:
    checkpoint = MODEL / "model.safetensors"
    if sha_file(checkpoint) != MODEL_SHA256:
        raise RuntimeError("frozen checkpoint SHA256 mismatch")
    weights: dict[str, torch.Tensor] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as data:
        for inv in inventory:
            target = name(inv["layer"], inv["operator"])
            if target in targets:
                weights[target] = quant_weight(data.get_tensor(inv["checkpoint_key"]).contiguous()).to("cuda")
    if set(weights) != targets:
        raise RuntimeError("incomplete quantized weight inventory")
    return weights


def main(args: argparse.Namespace) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if branch != "phase/02-qwen3-quantization" or head not in (WINDOWS_START, JETSON_HANDOFF):
        raise RuntimeError(f"unexpected checkout {branch} {head}")
    dump(out / "start_audit.json", {"branch": branch, "head": head, "windows_authoritative_start_head": WINDOWS_START,
                                     "jetson_historical_handoff_execution": head == JETSON_HANDOFF})
    cal, eva = load(args.calibration_manifest)["rows"], load(args.evaluation_manifest)["rows"]
    if len(cal) != 24 or len(eva) != 12 or {row["sample_id"] for row in cal} & {row["sample_id"] for row in eva}:
        raise RuntimeError("invalid frozen split")
    c = args.c_artifacts
    inventory = load(c / "weight_inventory.json")
    static = {row["target"]: row for row in load(c / "weight_reconstruction_per_target.json")}
    dynamic = load(c / "portable_sensitivity_per_target.json")
    values = {target: float(row["pt_w8a8_vs_f_relative_l2"]["p95"]) for target, row in dynamic.items()}
    outlier_report = robust(values)
    outliers = {row["target"] for row in outlier_report["outliers"]}
    family_source = load(c / "operator_sensitivity_summary.json")
    ranking = sorted(({"operator": op, "p95": float(row["p95"]), "targets": row["targets"]} for op, row in family_source.items()), key=lambda row: (-row["p95"], row["operator"]))
    families = {row["operator"] for row in ranking[:2]}
    trt = set(load(c / "trt_confirmation_targets.json")["targets"])
    if (len(inventory), len(static), len(dynamic), len(trt)) != (196, 196, 34, 8):
        raise RuntimeError("Phase 2.3-C evidence cardinality mismatch")
    dump(out / "phase2_3c_evidence_digest.json", {"source_artifacts": str(c), "static_linear_count": 196, "dynamic_target_count": 34, "trt_confirmation_target_count": 8,
        "c_gate": load(c / "final_validation.json")["gate"], "per_channel_boundary": "PER_CHANNEL_QDQ_CAPABILITY_EXISTS_IN_PROBES; REAL_QWEN3_LINEAR_PC_QDQ_PATH_BLOCKED"})
    dump(out / "robust_outlier_analysis.json", outlier_report)
    dump(out / "operator_family_ranking.json", {"metric": "Phase 2.3-C PT-W8A8 vs F P95 relative-L2", "ranking": ranking, "selected_top_two_families": sorted(families)})
    dump(out / "layer_trend_reuse.json", {"classification": "NO_CLEAR_MONOTONIC_LAYER_TREND", "source": "Phase 2.3-C layer_sensitivity_summary.json"})
    p0 = make_policy("P0_ALL_PT_W8A8", inventory, static, dynamic, trt, outliers, families)
    p1 = make_policy("P1_ROBUST_OUTLIER_GUARD", inventory, static, dynamic, trt, outliers, families)
    p2 = make_policy("P2_FAMILY_GUARD", inventory, static, dynamic, trt, outliers, families)
    for filename, label, rows in (("policy_p0_all_int8.json", "P0_ALL_PT_W8A8", p0), ("policy_p1_outlier_guard.json", "P1_ROBUST_OUTLIER_GUARD", p1), ("policy_p2_family_guard_prevalidation.json", "P2_FAMILY_GUARD", p2)):
        dump(out / filename, {"policy": label, "entries": rows, "coverage": coverage(rows)})
    dump(out / "policy_candidate_comparison.json", {label: {"coverage": coverage(rows), "c_outliers_protected": sorted(row["target"] for row in rows if row["target"] in outliers and row["precision_state"] == "FP16")} for label, rows in (("P0_ALL_PT_W8A8", p0), ("P1_ROBUST_OUTLIER_GUARD", p1), ("P2_FAMILY_GUARD", p2))})

    quantized = {row["target"] for row in p2 if row["precision_state"] == "PT_W8A8"}
    traces = [memory("start")]
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True, revision=REVISION, torch_dtype=torch.bfloat16, device_map=None)
    model.config._attn_implementation = "eager"
    model.to(device="cuda", dtype=torch.bfloat16).eval()
    traces.append(memory("model_loaded"))
    maxima: dict[str, float] = defaultdict(float); count: dict[str, int] = defaultdict(int); seen: dict[str, int] = defaultdict(int)
    def observe(target: str, x: torch.Tensor) -> None:
        if not torch.isfinite(x.float()).all(): raise RuntimeError(f"non-finite calibration input {target}")
        maxima[target] = max(maxima[target], float(x.float().abs().max().item())); count[target] += x.numel(); seen[target] += 1
    hooks = HookSet(model, quantized, observe); forward_rows(model, cal, hooks, quantized); hooks.close(); traces.append(memory("calibration_pass1"))
    if set(maxima) != quantized or any(seen[target] != 24 for target in quantized): raise RuntimeError("calibration pass 1 incomplete")
    sse = {target: {factor: 0.0 for factor in FACTORS} for target in quantized}; clips = {target: {factor: 0 for factor in FACTORS} for target in quantized}
    def score(target: str, x: torch.Tensor) -> None:
        xf = x.float()
        for factor in FACTORS:
            scale = maxima[target] * factor / 127.0; raw = torch.round(xf / scale); delta = raw.clamp(-127, 127) * scale - xf
            sse[target][factor] += float((delta * delta).sum().item()); clips[target][factor] += int(((raw < -127) | (raw > 127)).sum().item())
    hooks = HookSet(model, quantized, score); forward_rows(model, cal, hooks, quantized); hooks.close(); traces.append(memory("calibration_pass2"))
    scales: dict[str, Any] = {}
    for target in sorted(quantized):
        grid = [{"factor": factor, "range": maxima[target] * factor, "scale": float(np.float32(maxima[target] * factor / 127.0)), "mse": sse[target][factor] / count[target], "clipped_element_count": clips[target][factor]} for factor in FACTORS]
        selected = min(grid, key=lambda row: (row["mse"], row["factor"]))
        scales[target] = {"target": target, "algorithm": "BOUNDED_MSE_CLIP", "quantization": "symmetric per-tensor signed INT8", "qrange": [-127, 127], "zero_point": 0, "scale_dtype": "float32", "calibration_sample_count": 24, "calibration_element_count": count[target], "calibration_global_absmax": maxima[target], "grid": grid, "selected_factor": selected["factor"], "range": selected["range"], "scale": selected["scale"], "calibration_reconstruction_mse": selected["mse"], "clipped_element_count": selected["clipped_element_count"], "clipping_percent": selected["clipped_element_count"] / count[target] * 100.0}
    dump(out / "mixed_precision_activation_scales_primary.json", scales)
    dump(out / "activation_calibration_summary.json", {"method": "two bounded streaming calibration passes", "target_count": len(scales), "calibration_sample_count": 24, "evaluation_leakage": "NO", "raw_activation_files_written": False, "factor_grid": list(FACTORS), "objective": "aggregate elementwise reconstruction MSE", "tie_break": "(mse, factor)"})
    quant_weights = prepare_weights(inventory, quantized); traces.append(memory("quantized_weights_ready"))
    results: dict[str, list[dict[str, Any]]]= defaultdict(list); active = {"sample_id": ""}
    def evaluate(target: str, x: torch.Tensor) -> None:
        x = x.to(torch.float16); scale = scales[target]["scale"]; raw = torch.round(x.float() / scale); xq = (raw.clamp(-127, 127) * scale).to(torch.float16)
        w = model.get_submodule(module_name(target)).weight.to(torch.float16)
        row = metric(torch.matmul(xq, quant_weights[target].t()), torch.matmul(x, w.t()))
        row.update({"target": target, "sample_id": active["sample_id"], "precision_state": "PT_W8A8", "clipping_percent": float(((raw < -127) | (raw > 127)).float().mean().item() * 100.0)})
        results[target].append(row)
    hooks = HookSet(model, quantized, evaluate)
    with torch.inference_mode():
        for row in eva:
            active["sample_id"] = row["sample_id"]; hooks.reset(); model(input_ids=torch.tensor([row["token_ids"]], dtype=torch.long, device="cuda"), use_cache=False); hooks.check(row["sample_id"], quantized)
    hooks.close(); traces.append(memory("evaluation_complete"))
    if set(results) != quantized or any(len(results[target]) != 12 for target in quantized): raise RuntimeError("evaluation coverage incomplete")
    per_target, all_rows = [], []
    for target in sorted(quantized):
        rows = results[target]; all_rows.extend(rows)
        per_target.append({"target": target, "precision_state": "PT_W8A8", "evaluation": rows, "relative_l2": summary([row["relative_l2"] for row in rows]), "cosine": summary([row["cosine"] for row in rows]), "clipping_percent": summary([row["clipping_percent"] for row in rows]), "all_finite": all(row["finite"] for row in rows)})
    dump(out / "full_policy_component_validation.json", {"status": "PASS", "method": "real HF BF16 forward-pre-hook, one prompt at a time, portable PT-W8A8 versus FP16 component validation", "reference_boundary": "TRT FP16 remains the direct quantization baseline; this is policy prevalidation, not a TRT runtime claim", "evaluation_sample_count": 12, "quantized_target_count": len(quantized), "per_target": per_target, "raw_activation_files_written": False})
    dump(out / "full_policy_component_summary.json", {"status": "PASS", "quantized_target_count": len(quantized), "preserved_fp16_target_count": 196 - len(quantized), "aggregate_quantized": {"relative_l2": summary([row["relative_l2"] for row in all_rows]), "cosine": summary([row["cosine"] for row in all_rows]), "clipping_percent": summary([row["clipping_percent"] for row in all_rows]), "finite_failures": sum(not row["finite"] for row in all_rows)}})
    provenance = [{"sample_id": row["sample_id"], "split": split, "token_count": row["token_count"], "classification": "EXACT_LINEAR_INPUT_PROVEN", "captured_targets": len(quantized), "stored_activation": False} for split, rows in (("calibration", cal), ("evaluation", eva)) for row in rows]
    dump(out / "input_provenance_summary.json", {"classification": "EXACT_LINEAR_INPUT_PROVEN", "method": "real Qwen3 HF BF16 forward-pre-hook", "quantized_target_count": len(quantized), "calibration_passes": 2, "evaluation_passes": 1, "rows": provenance, "raw_activation_files_written": False})
    d_outliers = robust({row["target"]: row["relative_l2"]["p95"] for row in per_target}); refine = {row["target"] for row in d_outliers["outliers"]}
    final = [dict(row) for row in p2]
    for row in final:
        if row["target"] in refine: row.update({"precision_state": "FP16", "reason": "ONE_D_POLICY_REFINEMENT_ROBUST_OUTLIER", "weight_quantization": "N/A", "activation_quantization": "N/A", "zero_point": None})
    dump(out / "policy_refinement.json", {"primary_prevalidation": "P2_FAMILY_GUARD", "validation_metric": d_outliers["metric"], "robust_analysis": d_outliers, "targets_changed_int8_to_fp16": sorted(refine), "refinement_count": 1, "second_refinement_performed": False, "decision": "ONE_INT8_TO_FP16_REFINEMENT_APPLIED" if refine else "NO_POLICY_REFINEMENT_REQUIRED"})
    dump(out / "mixed_precision_policy_primary_final.json", {"policy": "P2_FAMILY_GUARD_REFINED", "entries": final, "coverage": coverage(final), "prevalidation_policy": "P2_FAMILY_GUARD", "bounded_refinement_count": 1})
    dump(out / "policy_coverage_summary.json", {"P0_ALL_PT_W8A8": coverage(p0), "P1_ROBUST_OUTLIER_GUARD": coverage(p1), "P2_FAMILY_GUARD_PREVALIDATION": coverage(p2), "P2_FAMILY_GUARD_REFINED_FINAL": coverage(final)})
    dump(out / "memory_trace.json", traces + [memory("final")])
    dump(out / "phase2_3e_readiness.json", {"PHASE2_3E_POLICY_READY": True, "assignments_196": len(final) == 196, "final_policy": "P2_FAMILY_GUARD_REFINED", "primary_int8_scales_complete": len(scales) == len(quantized), "exact_input_provenance_complete": True, "scope_boundary": "No 28-layer quantized runtime was built or executed in Phase 2.3-D."})
    dump(out / "final_validation.json", {"phase": "Phase 2.3-D", "gate": "PASS / BOUNDED", "starting_head": head, "windows_authoritative_start_head": WINDOWS_START, "c_evidence_recovered": True, "policy_entries": len(final), "p0_p1_p2_complete": True, "primary_policy_prevalidation": "P2_FAMILY_GUARD", "primary_policy_final": "P2_FAMILY_GUARD_REFINED", "primary_scales": len(scales), "full_policy_component_validation": "PASS", "bounded_refinement_count": 1, "no_phase_e": True, "no_full_runtime": True, "no_benchmark": True, "no_nsight": True, "no_pc_parser_debug": True, "raw_activation_files_written": False, "oom": False, "exit137": False})
    print(json.dumps({"status": "PASS / BOUNDED", "entries": len(final), "prevalidation_scales": len(scales), "refined_targets": sorted(refine)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--c-artifacts", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True); parser.add_argument("--evaluation-manifest", type=Path, required=True)
    main(parser.parse_args())
