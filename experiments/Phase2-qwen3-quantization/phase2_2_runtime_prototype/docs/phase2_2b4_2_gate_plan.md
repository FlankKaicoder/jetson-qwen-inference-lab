# B4.2 Gate Plan

The experiment is complete only when each gate has a repository artifact:

- **B4.2-1 Mapping:** 28 files, required tensor names/shapes/dtypes/bytes and SHA256 values match the frozen model identity.
- **B4.2-2 ONNX:** prefill and decode are opset 17, FP16-only, checker-valid, with node/initializer/operator counts recorded.
- **B4.2-3 TensorRT:** both graphs parse and build with the installed TensorRT; build memory and warnings are recorded.
- **B4.2-4 Prefill:** B=1,S=8 output and all 28 K/V tensors are shape-correct, finite, and CUDA-resident.
- **B4.2-5 Decode:** four one-token steps advance cache lengths 8→9→10→11→12 for every layer.
- **B4.2-6 Integrity:** old prefix is bitwise invariant, new slot is populated, and K/V pointers remain layer-isolated.
- **B4.2-7 Numerical propagation:** report max absolute error, RMSE, relative L2 and cosine for layers 0, 3, 7, 15 and 27 against the portable FP16 reference. Before execution, the bounded acceptance rule is defined as finite, shape-equal selected tensors, hidden/K/V cosine >= 0.99 and relative L2 <= 0.10, including Layer 27. Passing this rule permits `ACCEPTABLE_FOR_FULL_FP16_RUNTIME_STEP`; otherwise use `NUMERICAL_PROPAGATION_RISK_REQUIRES_REVIEW`. This engineering bound is not a full-model task-quality criterion.
- **B4.2-8 Resources:** classify measured host/device memory as `PASS`, `RISK`, or `BLOCKED`; theoretical capacity is never presented as measured capacity.

The primary architecture is attempted once. A fixed seven-way 4-layer partition is only a separately recorded fallback after a clear primary resource failure.
