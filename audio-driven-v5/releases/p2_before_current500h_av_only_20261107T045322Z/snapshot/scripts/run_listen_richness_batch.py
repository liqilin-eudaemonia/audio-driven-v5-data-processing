#!/usr/bin/env python3
"""Single explicitly assigned GPU: batched caption extraction with resource traces."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from listen_richness.common import digest, file_hash, read, rows, write, write_jsonl, safe_id
from listen_richness.extraction import contract, validate, compile_response, output_schema


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", required=True); p.add_argument("--output", required=True)
    p.add_argument("--task", choices=["extract", "score"], default="extract")
    p.add_argument("--gpu", type=int, required=True); p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--model", default="/private/ckpts/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--batch", type=int, default=256); p.add_argument("--max-batched-tokens", type=int, default=32768)
    p.add_argument("--memory", type=float, default=.92); p.add_argument("--attempts", type=int, default=2)
    a = p.parse_args()
    # Keep the extraction protocol/default untouched. Scoring is an independent sidecar task.
    if a.task == "score":
        from listen_richness_score import contract as make_contract, compile_response as compile_output, output_schema as schema_for
    else:
        make_contract, compile_output, schema_for = contract, compile_response, output_schema
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.gpu):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to the one explicitly assigned physical GPU")
    if not 1 <= a.batch <= 256 or not .1 <= a.memory <= .95 or not 1 <= a.attempts <= 5:
        raise ValueError("Invalid batch/memory/retry settings")
    query = ["nvidia-smi", "-i", str(a.gpu), "--query-gpu=uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
    initial = subprocess.check_output(query, text=True).strip().split(",")
    if initial[0].strip() != a.gpu_uuid or float(initial[1]) > 1024:
        raise RuntimeError("Assigned GPU identity differs or no longer idle; do not overlay another job")
    requests = list(rows(a.requests)); ids = [safe_id(r["sample_id"]) for r in requests]
    if not requests or len(set(ids)) != len(ids): raise ValueError("Empty/duplicate requests")
    c = make_contract(a.model)
    output_tokens = c["max_tokens"]
    captions = {}
    for r in requests:
        if r["extraction_contract"] != c: raise ValueError("Model/prompt contract mismatch")
        captions[r["sample_id"]] = json.loads(r["messages"][1]["content"])["caption"]
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError("Use a fresh run output directory")
    graph_sizes = [n for n in (1,2,4,8,16,32,64,128,256) if n <= a.batch]
    if graph_sizes[-1] != a.batch: graph_sizes.append(a.batch)
    model = Path(a.model)
    protocol = dict(schema="listen-richness-batch-v1", task=a.task, input_sha256=file_hash(a.requests),
                    extraction_contract=c, physical_gpu=a.gpu, gpu_uuid=a.gpu_uuid,
                    batch=a.batch, max_num_seqs=a.batch, max_num_batched_tokens=a.max_batched_tokens,
                    gpu_memory_utilization=a.memory, max_model_len=12288,
                    compilation=dict(mode=0,cudagraph_mode="FULL_DECODE_ONLY",cudagraph_capture_sizes=graph_sizes,
                                     max_cudagraph_capture_size=graph_sizes[-1]),
                    vllm=version("vllm"), torch=version("torch"),
                    code={str(f.relative_to(ROOT)):file_hash(f) for f in [Path(__file__),ROOT/"src/listen_richness/extraction.py",ROOT/"scripts/listen_richness_score.py"]},
                    model_files={f.name:dict(size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns)
                                 for f in model.iterdir() if f.suffix in (".json", ".safetensors")},
                    input_count=len(requests),started_at=time.time())
    write(out/"protocol.json",protocol)
    stage = ["loading"]; stop = threading.Event()
    def monitor():
        with (out/"resources.jsonl").open("w", buffering=1) as log:
            while not stop.is_set():
                now = time.time()
                try:
                    fields = [x.strip() for x in subprocess.check_output(query,text=True,timeout=5).strip().split(",")]
                    entry=dict(at=now,stage=stage[0],gpu_uuid=fields[0],memory_used_mib=float(fields[1]),
                               memory_total_mib=float(fields[2]),gpu_utilization=float(fields[3]),
                               host_loadavg=os.getloadavg())
                except Exception as exc: entry=dict(at=now,stage=stage[0],error=str(exc))
                log.write(json.dumps(entry)+"\n"); stop.wait(1)
    watcher = threading.Thread(target=monitor,daemon=True); watcher.start()
    accepted = {}; failures = {}; rounds = []; feedback = {}; start = time.monotonic()
    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model,local_files_only=True)
        llm = LLM(model=a.model,dtype="bfloat16",tensor_parallel_size=1,
                  gpu_memory_utilization=a.memory,max_model_len=12288,max_num_seqs=a.batch,
                  max_num_batched_tokens=a.max_batched_tokens,enable_chunked_prefill=True,
                  enable_prefix_caching=True,enforce_eager=False,seed=20261031,
                  compilation_config=protocol["compilation"],
                  structured_outputs_config={"backend":"xgrammar","disable_any_whitespace":True})
        load_s=time.monotonic()-start; pending=requests
        for attempt in range(a.attempts):
            if not pending: break
            retry=[]; stage[0]=f"generation_attempt_{attempt+1}"
            for offset in range(0,len(pending),a.batch):
                batch=pending[offset:offset+a.batch]; prompts=[]
                for r in batch:
                    messages=list(r["messages"])
                    if r["sample_id"] in feedback:
                        messages.append(dict(role="user",content="Previous attempt failed validation: "+feedback[r["sample_id"]]+". Return a fresh extraction from the original caption."))
                    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    if len(tokenizer.encode(text))+output_tokens>12288: raise ValueError("Context overflow; no silent truncation")
                    prompts.append(text)
                sampling=[SamplingParams(temperature=0,max_tokens=output_tokens,
                          structured_outputs=StructuredOutputsParams(json=schema_for(captions[r['sample_id']]))) for r in batch]
                t=time.monotonic(); outputs=llm.generate(prompts,sampling,use_tqdm=True); elapsed=time.monotonic()-t
                if len(outputs)!=len(batch): raise RuntimeError("Missing engine outputs")
                tokens=0; good=0
                for r,resp,prompt in zip(batch,outputs,prompts):
                    ans=resp.outputs[0]; sid=r["sample_id"]; tokens+=len(ans.token_ids)
                    entry=dict(sample_id=sid,attempt=attempt+1,raw=ans.text,finish_reason=ans.finish_reason,
                               completion_tokens=len(ans.token_ids),prompt_tokens=len(resp.prompt_token_ids),
                               rendered_prompt=prompt,snapshot_sha256=r["snapshot_sha256"])
                    try:
                        if ans.finish_reason!="stop": raise ValueError("Incomplete model generation")
                        extraction=compile_output(json.loads(ans.text),captions[sid])
                        accepted[sid]=dict(sample_id=sid,snapshot_sha256=r["snapshot_sha256"],
                                          extraction_contract=c,finish_reason="stop",**{"score" if a.task=="score" else "extraction":extraction})
                        failures.pop(sid,None); good+=1
                    except Exception as exc:
                        entry["error"]=str(exc);feedback[sid]=str(exc);failures[sid]=str(exc);retry.append(r)
                    write(out/"attempts"/f"{sid}.{attempt+1}.json",entry)
                metric=dict(attempt=attempt+1,batch=len(batch),seconds=elapsed,accepted=good,
                            output_tokens=tokens,output_tokens_per_second=tokens/elapsed,
                            clips_per_second=len(batch)/elapsed)
                rounds.append(metric);write(out/"progress.json",dict(accepted=len(accepted),failed=len(failures),rounds=rounds))
                print(json.dumps(metric),flush=True)
            pending=retry
        stage[0]="finished"
        write_jsonl(out/"responses.jsonl",(accepted[k] for k in sorted(accepted)))
        summary=dict(input_count=len(requests),accepted=len(accepted),failed=len(failures),failures=failures,
                     load_seconds=load_s,total_seconds=time.monotonic()-start,rounds=rounds,
                     model_quality_not_human_verified=True)
        write(out/"summary.json",summary)
    finally:
        stop.set();watcher.join(timeout=6)
    if failures: return 1
    return 0


if __name__=="__main__": raise SystemExit(main())
