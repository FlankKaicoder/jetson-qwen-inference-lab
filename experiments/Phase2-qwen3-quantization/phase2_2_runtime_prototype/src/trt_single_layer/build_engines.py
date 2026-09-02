import argparse
from pathlib import Path
import onnx
import tensorrt as trt

def build(onnx_path: Path, engine_path: Path, kind: str):
    logger=trt.Logger(trt.Logger.WARNING); builder=trt.Builder(logger)
    network=builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)); model=onnx.load(str(onnx_path)); parser=trt.OnnxParser(network,logger)
    ok=parser.parse(model.SerializeToString()); errors=[str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not ok: raise RuntimeError(f'{kind} parse failed: {errors}')
    cfg=builder.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16); profile=builder.create_optimization_profile()
    if kind=='prefill':
        profile.set_shape('hidden_states',(1,1,1024),(1,8,1024),(1,16,1024)); profile.set_shape('position_ids',(1,1),(1,8),(1,16))
    else:
        profile.set_shape('hidden_states',(1,1,1024),(1,1,1024),(1,1,1024)); profile.set_shape('position_ids',(1,1),(1,1),(1,1)); profile.set_shape('past_k',(1,8,1,128),(1,8,8,128),(1,8,16,128)); profile.set_shape('past_v',(1,8,1,128),(1,8,8,128),(1,8,16,128))
    cfg.add_optimization_profile(profile); blob=builder.build_serialized_network(network,cfg)
    if blob is None: raise RuntimeError(f'{kind} build returned None')
    engine_path.parent.mkdir(parents=True,exist_ok=True); engine_path.write_bytes(bytes(blob)); return {'parse':'PASS','build':'PASS','bytes':engine_path.stat().st_size,'parser_errors':errors}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prefill-onnx',type=Path,required=True); p.add_argument('--decode-onnx',type=Path,required=True); p.add_argument('--prefill-engine',type=Path,required=True); p.add_argument('--decode-engine',type=Path,required=True); p.add_argument('--result',type=Path,required=True); a=p.parse_args()
    r={'prefill':build(a.prefill_onnx,a.prefill_engine,'prefill'),'decode':build(a.decode_onnx,a.decode_engine,'decode')}; a.result.parent.mkdir(parents=True,exist_ok=True); a.result.write_text(__import__('json').dumps(r,indent=2)+'\n'); print(r)
if __name__=='__main__': main()
