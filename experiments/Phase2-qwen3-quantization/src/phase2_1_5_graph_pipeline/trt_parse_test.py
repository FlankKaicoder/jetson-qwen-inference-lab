import argparse, json
from pathlib import Path
import onnx
import tensorrt as trt
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--onnx',type=Path,required=True); p.add_argument('--engine',type=Path,required=True); p.add_argument('--shape',default='1,1024'); a=p.parse_args()
    model=onnx.load(a.onnx); onnx.checker.check_model(model)
    ops=[n.op_type for n in model.graph.node]; g={'onnx':'PASS','node_count':len(ops),'operators':ops,'inputs':[i.name for i in model.graph.input],'outputs':[o.name for o in model.graph.output]}
    log=trt.Logger(trt.Logger.WARNING); b=trt.Builder(log); net=b.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)); parser=trt.OnnxParser(net,log)
    ok=parser.parse(model.SerializeToString()); errors=[str(parser.get_error(i)) for i in range(parser.num_errors)]
    g.update({'trt_parse':'PASS' if ok else 'FAIL','parser_errors':errors})
    if ok:
        cfg=b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16); prof=b.create_optimization_profile(); prof.set_shape('input',(1,1024),(32,1024),(32,1024)); cfg.add_optimization_profile(prof); blob=b.build_serialized_network(net,cfg)
        if blob is None: g['engine']='FAIL'
        else:
            a.engine.parent.mkdir(parents=True,exist_ok=True); a.engine.write_bytes(bytes(blob)); g['engine']='PASS'; g['engine_bytes']=a.engine.stat().st_size
            rt=trt.Runtime(log); e=rt.deserialize_cuda_engine(bytes(blob)); c=e.create_execution_context(); m=int(a.shape.split(',')[0]); c.set_input_shape('input',(m,1024)); x=torch.randn((m,1024),device='cuda',dtype=torch.float16)
            out_shape=tuple(c.get_tensor_shape('output')); out_dtype=e.get_tensor_dtype('output')
            dtype_map={trt.DataType.FLOAT:torch.float32,trt.DataType.HALF:torch.float16,trt.DataType.BF16:torch.bfloat16}
            if out_dtype not in dtype_map: raise RuntimeError(f'unsupported output dtype: {out_dtype}')
            y=torch.empty(out_shape,device='cuda',dtype=dtype_map[out_dtype]); c.set_tensor_address('input',x.data_ptr()); c.set_tensor_address('output',y.data_ptr()); g['gpu_execute']=bool(c.execute_async_v3(torch.cuda.current_stream().cuda_stream)); torch.cuda.synchronize(); g.update({'output_shape':list(out_shape),'output_dtype':str(out_dtype),'output_finite':bool(torch.isfinite(y).all().item())})
    print(json.dumps(g,sort_keys=True))
if __name__=='__main__': main()
