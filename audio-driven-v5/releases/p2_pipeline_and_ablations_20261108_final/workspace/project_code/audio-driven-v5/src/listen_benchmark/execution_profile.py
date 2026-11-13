"""Validated per-GPU execution trials shared by worker and supervisor."""
from copy import deepcopy
import math

EXECUTION_KEYS=('batch_size','tokens','frames','memory','eager','decode_graph_capture_sizes','renderer_workers','decode_backend','rolling_annotation_groups')


def execution_for_gpu(selected,config,gpu):
    if type(gpu) is not int or not 0<=gpu<8:raise ValueError('GPU index must be 0..7')
    overrides=config.get('gpu_execution_overrides',{})
    if not isinstance(overrides,dict) or set(overrides)-{str(i) for i in range(8)}:
        raise ValueError('Execution overrides must name node17 GPU indices 0..7')
    override=overrides.get(str(gpu),{})
    if not isinstance(override,dict) or set(override)-{'memory','eager','decode_graph_capture_sizes','renderer_workers','decode_backend','rolling_annotation_groups'}:
        raise ValueError('Execution trial may only override memory, decode graph and CPU renderer settings')
    result=deepcopy(selected)
    rolling_enabled=config.get('rolling_inference',False) and (
        'rolling_canary_gpus' not in config or gpu in config['rolling_canary_gpus'])
    if 'rolling_inference' in config:
        result['rolling_annotation_groups']=2 if rolling_enabled else 1
    if 'rolling_annotation_groups' in override and not rolling_enabled:
        raise ValueError('Group capacity override requires rolling inference on this GPU')
    if 'renderer_workers' in config:
        result['renderer_workers']=config['renderer_workers']
        if 'renderer_canary_gpus' in config and gpu not in config['renderer_canary_gpus']:
            result['renderer_workers']=0
    result.update(deepcopy(override))
    if 'rolling_annotation_groups' in result:
        groups=result['rolling_annotation_groups']
        if type(groups) is not int or groups not in ((2,4) if rolling_enabled else (1,)):
            raise ValueError('Rolling annotation capacity must be two or four')
    if 'decode_backend' in result and result['decode_backend'] not in ('thread','process','hybrid'):
        raise ValueError('Unknown CPU decode backend')
    if 'renderer_workers' in result and (type(result['renderer_workers']) is not int or result['renderer_workers'] not in (0,2,4)):
        raise ValueError('CPU renderer trial supports disabled, two or four workers')
    memory=result['memory']
    if isinstance(memory,bool) or not isinstance(memory,(float,int)) or not math.isfinite(memory) or not 0<memory<1:
        raise ValueError('Invalid GPU memory fraction')
    if not isinstance(result.get('eager'),bool):raise ValueError('Explicit eager boolean required')
    sizes=result.get('decode_graph_capture_sizes')
    if sizes is None:
        if not result['eager']:raise ValueError('Non-eager trial requires explicit decode capture sizes')
    else:
        if result['eager'] or not isinstance(sizes,list) or not sizes:
            raise ValueError('Decode graph sizes require non-eager execution')
        if any(type(s) is not int or s<1 for s in sizes) or sizes!=sorted(set(sizes)):
            raise ValueError('Decode graph sizes must be positive increasing integers')
        if sizes[-1]>result['batch_size']:raise ValueError('Decode graph size exceeds sequence limit')
    return result


def engine_graph_kwargs(execution):
    sizes=execution.get('decode_graph_capture_sizes')
    result={'enforce_eager':execution['eager']}
    if sizes is not None:
        result['compilation_config']={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY',
            'cudagraph_capture_sizes':list(sizes),'max_cudagraph_capture_size':sizes[-1]}
    return result
