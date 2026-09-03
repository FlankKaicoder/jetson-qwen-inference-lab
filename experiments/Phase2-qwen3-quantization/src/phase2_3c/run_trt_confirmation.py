from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

def metric(a,b):
    af,bf=a.float(),b.float(); d=af-bf; den=torch.linalg.vector_norm(bf).item()
    return {"shape_equal":list(a.shape)==list(b.shape),"finite":bool(torch.isfinite(af).all() and torch.isfinite(bf).all()),"max_abs":float(d.abs().max()),"mean_abs":float(d.abs().mean()),"rmse":float(torch.sqrt(torch.mean(d*d))),"relative_l2":float(torch.linalg.vector_norm(d)/den) if den else 0.0,"cosine":float(torch.nn.functional.cosine_similarity(af.reshape(1,-1),bf.reshape(1,-1)))}

def make_onnx(path, mode, w, scales, a_scale):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    nodes=[]; init=[numpy_helper.from_array(w.astype(np.float16) if mode == "fp16" else w.astype(np.int8),name="weight")]; wi="weight"; xi="input"
    if mode != "fp16":
        init += [numpy_helper.from_array(scales.astype(np.float16),name="weight_scale"),numpy_helper.from_array(np.asarray(0,dtype=np.int8),name="weight_zero_point")]
        nodes.append(helper.make_node("DequantizeLinear",["weight","weight_scale","weight_zero_point"],["weight_dq"],axis=1)); wi="weight_dq"
    if mode.endswith("a8"):
        init += [numpy_helper.from_array(np.asarray(a_scale,dtype=np.float16),name="activation_scale"),numpy_helper.from_array(np.asarray(0,dtype=np.int8),name="activation_zero_point")]
        nodes += [helper.make_node("QuantizeLinear",["input","activation_scale","activation_zero_point"],["input_q"]),helper.make_node("DequantizeLinear",["input_q","activation_scale","activation_zero_point"],["input_dq"])]
        xi="input_dq"
    nodes.append(helper.make_node("MatMul",[xi,wi],["output"],name="target_linear_matmul"))
    graph=helper.make_graph(nodes,"phase2_3c",[helper.make_tensor_value_info("input",TensorProto.FLOAT16,[1,"seq",int(w.shape[0])])],[helper.make_tensor_value_info("output",TensorProto.FLOAT16,[1,"seq",int(w.shape[1])])],init)
    m=helper.make_model(graph,opset_imports=[helper.make_opsetid("",17)],producer_name="phase2_3c"); m.ir_version=9; onnx.checker.check_model(m); path.write_bytes(m.SerializeToString())

class Runtime:
    def __init__(self,p):
        import tensorrt as trt
        self.trt=trt; self.engine=trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(p.read_bytes()); self.ctx=self.engine.create_execution_context(); self.stream=torch.cuda.current_stream()
    def run(self,x):
        self.ctx.set_input_shape("input",tuple(x.shape)); self.ctx.set_tensor_address("input",x.data_ptr()); out=None
        for i in range(self.engine.num_io_tensors):
            n=self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n)==self.trt.TensorIOMode.OUTPUT:
                out=torch.empty(tuple(self.ctx.get_tensor_shape(n)),device="cuda",dtype=torch.float16); self.ctx.set_tensor_address(n,out.data_ptr())
        if out is None or not self.ctx.execute_async_v3(self.stream.cuda_stream): raise RuntimeError("execute failed")
        self.stream.synchronize(); return out

def main(a):
    import tensorrt as trt
    out,work=a.out,a.work; out.mkdir(parents=True,exist_ok=False); work.mkdir(parents=True,exist_ok=False)
    payload=torch.load(a.payload,map_location="cpu",weights_only=True); eva=json.loads(a.evaluation.read_text())["rows"]
    results=[]; summaries={}; inspectors={}
    for name,data in payload["weights"].items():
        w={k:v.numpy() for k,v in data.items()}; scale=float(payload["scales"][name]["scale"]); modes=[("fp16",w["fp16"].T,np.asarray([1],np.float16),None),("pt_w8",w["pt_w8"].T,np.asarray([float(payload["scales"][name].get("weight_scale",1))],np.float16),None),("pc_w8",w["pc_w8"].T,np.ones((1,w["pc_w8"].shape[0]),np.float16),None),("pt_w8a8",w["pt_w8"].T,np.asarray([float(payload["scales"][name].get("weight_scale",1))],np.float16),scale),("pc_w8a8",w["pc_w8"].T,np.ones((1,w["pc_w8"].shape[0]),np.float16),scale)]
        # derive exact QDQ scales from original matrices
        wf=data["fp16"].float(); pt_scale=float(wf.abs().max()/127.0); qpt=torch.round(wf/pt_scale).clamp(-127,127).to(torch.int8).numpy(); pc_scale=(wf.abs().amax(dim=1,keepdim=True)/127.0); pc_scale=torch.where(pc_scale==0,torch.ones_like(pc_scale),pc_scale); qpc=torch.round(wf/pc_scale).clamp(-127,127).to(torch.int8).numpy(); pc_scale=pc_scale.reshape(-1).numpy().astype(np.float16)
        modes=[("fp16",data["fp16"].T.numpy(),np.asarray([1],np.float16),None),("pt_w8",qpt.T,np.asarray([pt_scale],np.float16),None),("pc_w8",qpc.T,pc_scale,None),("pt_w8a8",qpt.T,np.asarray([pt_scale],np.float16),scale),("pc_w8a8",qpc.T,pc_scale,scale)]
        rt={}; mode_status={}
        for mode,arr,ss,aa in modes:
            try:
                p=work/(name.replace(':','_')+'_'+mode+'.onnx'); e=work/(name.replace(':','_')+'_'+mode+'.engine'); make_onnx(p,mode,arr,ss,aa); b=trt.Builder(trt.Logger(trt.Logger.WARNING)); n=b.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)); parser=trt.OnnxParser(n,trt.Logger(trt.Logger.WARNING));
                if not parser.parse(p.read_bytes()): raise RuntimeError('parse failed')
                c=b.create_builder_config(); c.set_flag(trt.BuilderFlag.FP16); c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,256<<20); prof=b.create_optimization_profile(); prof.set_shape("input",(1,1,int(arr.shape[0])),(1,64,int(arr.shape[0])),(1,256,int(arr.shape[0]))); c.add_optimization_profile(prof); blob=b.build_serialized_network(n,c)
                if blob is None: raise RuntimeError('build failed')
                e.write_bytes(bytes(blob)); rt[mode]=Runtime(e); eng=trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(bytes(blob)); inspectors[name+':'+mode]=eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON); mode_status[mode]="PASS"
            except Exception as exc:
                mode_status[mode]={"status":"BLOCKED","error":f"{type(exc).__name__}: {exc}"}; inspectors[name+':'+mode]=mode_status[mode]
        rows=[]
        for row in eva:
            x=payload["captured"][name][row["sample_id"]].to(device="cuda",dtype=torch.float16); y={m:rt[m].run(x) for m in rt}; mm={}
            pairs=(("trt_pt_w8_vs_trt_fp16","pt_w8","fp16"),("trt_pc_w8_vs_trt_fp16","pc_w8","fp16"),("trt_pt_w8a8_vs_trt_pt_w8","pt_w8a8","pt_w8"),("trt_pc_w8a8_vs_trt_pc_w8","pc_w8a8","pc_w8"),("trt_pt_w8a8_vs_trt_fp16","pt_w8a8","fp16"),("trt_pc_w8a8_vs_trt_fp16","pc_w8a8","fp16"))
            for key,u,v in pairs:
                if u in y and v in y: mm[key]=metric(y[u],y[v])
            rows.append({"target":name,"sample_id":row["sample_id"],"metrics":mm})
        results.extend(rows); summaries[name]={"evaluation":rows,"mode_status":mode_status,"all_outputs_finite":all(v["finite"] for r in rows for v in r["metrics"].values())}; print('trt',name,flush=True)
    (out/'trt_confirmation_results.json').write_text(json.dumps(results,indent=2)+'\n'); (out/'trt_engine_summary.json').write_text(json.dumps(summaries,indent=2)+'\n'); (out/'trt_inspector_summary.json').write_text(json.dumps(inspectors,indent=2)+'\n'); print(json.dumps({'status':'PASS','targets':len(summaries)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--payload',type=Path,required=True); p.add_argument('--evaluation',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--work',type=Path,required=True); main(p.parse_args())
