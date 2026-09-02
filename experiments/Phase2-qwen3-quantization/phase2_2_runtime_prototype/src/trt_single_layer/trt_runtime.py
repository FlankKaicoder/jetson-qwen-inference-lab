from __future__ import annotations
import tensorrt as trt
import torch

class TRTRuntime:
    """Binds CUDA tensors directly; cache payload never crosses the host."""
    def __init__(self, engine_path):
        self.logger=trt.Logger(trt.Logger.WARNING); self.engine=trt.Runtime(self.logger).deserialize_cuda_engine(open(engine_path,'rb').read()); self.context=self.engine.create_execution_context(); self.stream=torch.cuda.current_stream()
    def execute(self, inputs):
        for name,tensor in inputs.items(): self.context.set_input_shape(name,tuple(tensor.shape)); self.context.set_tensor_address(name,tensor.data_ptr())
        outputs={}
        for i in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name)==trt.TensorIOMode.OUTPUT:
                shape=tuple(self.context.get_tensor_shape(name)); dtype={trt.DataType.HALF:torch.float16,trt.DataType.FLOAT:torch.float32}[self.engine.get_tensor_dtype(name)]; outputs[name]=torch.empty(shape,device='cuda',dtype=dtype); self.context.set_tensor_address(name,outputs[name].data_ptr())
        ok=bool(self.context.execute_async_v3(self.stream.cuda_stream)); self.stream.synchronize()
        if not ok: raise RuntimeError('TensorRT execute_async_v3 returned false')
        return outputs
