"""Monitor only this run, retry its failed stages, merge official metrics and report."""
import argparse,copy,csv,fcntl,importlib.util,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'compare/deployment'),str(ROOT/'scripts')]
from vico_more_common import read,save,sha
SOURCE=ROOT/'experiments/p2_l2l_public_renderer_20261110'
JOBS=ROOT/'experiments/p2_vico_additional_baselines_20261107/jobs'
NAMES=['conan','fallon','stephen','trevor']


def stages():
    result=[]
    for name in [*NAMES,'listenformer']:
        script=OUT/('run_listenformer_emoca.py' if name=='listenformer' else 'run_emoca.py')
        result.append(dict(name='emoca_'+name,job='vico_20261111_emoca_'+name,
            python=str(ROOT/'compare/.envs/vico_more/bin/python'),script=str(script),args=[] if name=='listenformer' else [name],
            marker=str(OUT/'emoca'/name/'EXTRACTION_COMPLETE.json')))
    for name in NAMES:
        result.append(dict(name='image_'+name,job='vico_20261111_image_'+name+'_evalenv',
            python=str(ROOT/'compare/.envs/vico_eval/bin/python'),script=str(SOURCE/'image_metrics.py'),
            args=['--work',str(SOURCE/'image_eval'/name)],marker=str(SOURCE/'image_eval'/name/'COMPLETE.json')))
    return result


def progress():
    rows=[]
    for s in stages():
        done=Path(s['marker']).exists();row=dict(stage=s['name'],complete=done)
        if s['name'].startswith('emoca_'):
            folder=OUT/'emoca'/s['name'][6:];status=folder/'status.json'
            row.update(read(status) if status.exists() else dict(completed=0,total=43))
            row['stage']=s['name'];row['complete']=done
        else:
            folder=SOURCE/'image_eval'/s['name'][6:]/'metrics'
            row['metric_receipts']=len(list(folder.glob('*.json')))
            row['metric_receipts_total']=46
        rows.append(row)
    state=dict(at=time.time(),state='running',gpu='node11 physical GPU7',stages=rows,
        emoca_complete_clips=sum(r.get('completed',0) for r in rows if r['stage'].startswith('emoca_')),
        emoca_total_clips=215,excluded=['Avatar Forcing'])
    save(OUT/'STATUS.json',state);return state


def merge_and_score():
    folder=OUT/'unified_emoca';folder.mkdir(exist_ok=True)
    p=copy.deepcopy(read(SOURCE/'emoca_eval/protocol.json'))
    p['methods']=['joint','joint_sft2752','dystream','listenformer',*['l2l_gpavatar_'+n for n in NAMES]]
    p['reused_feature_records']=[r for r in p['reused_feature_records'] if r['role']!='avatarforcing_audio_only']
    p['reused_feature_records']+=read(OUT/'emoca/listenformer/EXTRACTION_COMPLETE.json')['records']
    records=[];shards={}
    for name in [*NAMES,'listenformer']:
        base=OUT/'emoca'/name;complete=read(base/'EXTRACTION_COMPLETE.json');spec=read(base/'EXTRACTION_PROTOCOL.json')
        assert complete['success'] and len(complete['records'])==43
        assert spec['source_protocol_sha256']==sha(base/'protocol.json')
        for path,h in read(base/'protocol.json')['source_files'].items():assert sha(path)==h,path
        assert complete['protocol_sha256']==sha(base/'EXTRACTION_PROTOCOL.json')
        for r in complete['records']:
            assert sha(r['path'])==r['sha256'] and r['extraction_protocol_sha256']==complete['protocol_sha256']
        shards[name]={str(base/f):sha(base/f) for f in ['protocol.json','EXTRACTION_PROTOCOL.json','EXTRACTION_COMPLETE.json']}
        if name in NAMES:records.extend(complete['records'])
    p['shards']=shards;p['excluded_methods']=['avatarforcing_audio_only']
    p['source_files'][str(Path(__file__).resolve())]=sha(__file__)
    save(folder/'protocol.json',p)
    save(folder/'EXTRACTION_COMPLETE.json',dict(success=True,tasks=172,records=records,shards=shards))
    spec=importlib.util.spec_from_file_location('unchanged_gp_evaluator',SOURCE/'evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=folder;module.METHODS=p['methods'];module.score()


def reports():
    motion=read(OUT/'unified_emoca/RESULTS.json')
    shutil.copyfile(OUT/'unified_emoca/metrics.csv',OUT/'unified_motion_metrics.csv')
    sources={'ListenFormer (local retraining)':ROOT/'experiments/p2_vico_additional_baselines_20261107/listenformer/RESULTS.json',
             'DyStream':ROOT/'experiments/p2_compare_vico43_20261106/dystream/IMAGE_RESULTS.json'}
    sources.update({'L2L '+n+' + GPAvatar':SOURCE/'image_eval'/n/'RESULTS.json' for n in NAMES})
    images=[]
    for name,path in sources.items():
        r=read(path);assert r['clips']==43
        if name.startswith('L2L'):
            done=read(path.parent/'COMPLETE.json');assert done['passed'] and done['results_sha256']==sha(path)
        images.append(dict(method=name,clips=r['clips'],frames=r['frames'],**r['metrics'],source=str(path),source_sha256=sha(path)))
    with (OUT/'image_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['method','clips','frames','psnr','ssim','fid','csim','cpbd','source','source_sha256']);writer.writeheader();writer.writerows(images)
    original=ROOT/'experiments/p2_vico_report_alignment_20261110/report_vs_local_all.csv'
    rows=list(csv.DictReader(original.open()));columns=list(rows[0])
    for name,result in motion['methods'].items():
        for metric in ['FD','P_FD','MSE','SID','Var','rPCC']:
            for group in ['exp','pose']:
                reference=next((r for r in rows if r['method']=='L2L conan' and r['metric']==metric and r['group']==group),None) if name.startswith('l2l_gpavatar') else None
                row={c:'' for c in columns}
                row.update(method=name+' (common EMOCA 20261111)',metric=metric,group=group,local=result['metrics'][metric][group],
                    reported=reference['reported'] if reference else '',reported_dataset='ViCo' if reference else '',
                    local_dataset='ViCo test43',reported_space='EMOCA' if reference else '',local_space='EMOCA-v2 RGB readout',
                    comparability='different_checkpoint_and_renderer' if reference else 'local_common_readout_only',
                    source='DIM Table1' if reference else '',notes='Current common finite-frame mask excludes Avatar Forcing; rows with native DECA/BFM stay separate')
                rows.append(row)
    with (OUT/'report_vs_local_all.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    lines=['# ViCo 已实现方法全量评测（2026-11-11）','',
        '已完成。使用 node11 物理GPU7，既有正确推理结果经SHA核验后复用。本轮新增215条EMOCA提取（L2L四套权重各43条、ListenFormer43条），以及L2L四套权重各43条视频的五项图像指标。Avatar Forcing不运行，也不参与本轮共同有效帧掩码。','',
        '## 图像指标','', '| 方法 | PSNR ↑ | SSIM ↑ | FID ↓ | CSIM ↑ | CPBD ↑ |','|---|---:|---:|---:|---:|---:|']
    for r in images:lines.append('| '+r['method']+' | '+' | '.join(f'{r[k]:.6f}' for k in ['psnr','ssim','fid','csim','cpbd'])+' |')
    lines+=['','## 统一EMOCA动作指标','',f"43条，共同有效评分帧 {motion['shared_scored_frames']}。speaker偏移一帧，所有方法使用相同有效帧交集；官方DIM指标函数及既定逐系数方差口径保持不变。",'',
        '| 方法 | FD Exp ↓ | FD Pose ↓ | P-FD Exp ↓ | P-FD Pose ↓ | MSE Exp ↓ | MSE Pose ↓ |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in motion['methods'].items():lines.append('| '+name+' | '+' | '.join(f'{r["metrics"][m][g]:.6f}' for m in ['FD','P_FD','MSE'] for g in ['exp','pose'])+' |')
    lines+=['','## 文件和适用范围','',
        '- [完整动作表](unified_motion_metrics.csv)：还包含SID、Var、rPCC。[逐条覆盖与原始数值](unified_emoca/RESULTS.json)。',
        '- [完整图像表](image_metrics.csv)。ListenFormer和DyStream图像成绩复用此前完整评测；四行L2L+GPAvatar为本轮新测。',
        '- [论文report与本地指标对照](report_vs_local_all.csv)：保留此前原生DECA/BFM行，另加本轮统一RGB→EMOCA行，不能混为一个表示空间。',
        '- DualTalk官方权重43条动作输出及原生指标已完成，尚无经过确认的RGB渲染/全局头部映射，图像指标与统一RGB→EMOCA指标为N/A；原生成绩见上表链接中的DualTalk行。',
        '- ListenFormer使用此前完整200轮本地重训权重；没有把它标成作者公开checkpoint。L2L使用四套公开人物权重和GPAvatar公开渲染器，均非DIM论文中的ViCo重训L2L，仍不能宣称复现该论文数值。',
        '- 本轮共同掩码排除了Avatar Forcing、加入ListenFormer，历史DyStream/自有模型数值可能随共同覆盖改变，须使用本轮同表比较。',
        '- 仍缺少可用权重/代码的方法见[待测清单](../p2_vico_additional_optimization_20261110/PENDING_METHODS.csv)，没有虚构推理结果或重新启动训练。','',
        '## 执行与依赖','',
        'EMOCA沿用原官方编码和裁脸路径，FP32、batch16，按人物分片。仅在本任务进程中禁用不使用的wandb日志模块，避免共享环境wandb/protobuf冲突；不修改共享包。图像指标使用原vico_eval环境和官方脚本，FID/CSIM batch64。各worker保留断点回执；监督器只重试本次失败任务，未停止其他任务。','']
    (OUT/'README.md').write_text('\n'.join(lines))
    archive=ROOT/'experiments/p2_vico_report_alignment_20261110';seal=read(archive/'COMPLETE.json')
    assert all(sha(archive/p)==h for p,h in seal['files'].items())
    save(OUT/'COMPLETE.json',dict(success=True,emoca_clips=215,image_scored_clips=172,original_metric_files_unchanged=len(seal['files']),
        excluded=['Avatar Forcing'],files={str(OUT/n):sha(OUT/n) for n in ['README.md','image_metrics.csv','unified_motion_metrics.csv','report_vs_local_all.csv','unified_emoca/COMPLETE.json','run_emoca.py','run_listenformer_emoca.py','finalize.py']}))


def watch():
    state_path=OUT/'RETRIES.json';retries=read(state_path) if state_path.exists() else {}
    children={};handles=[]
    while True:
        state=progress()
        if all(r['complete'] for r in state['stages']):break
        for s in stages():
            if Path(s['marker']).exists():continue
            name=s['name'];count=retries.get(name,0)
            if name in children:
                code=children[name].poll()
                if code is None:continue
                children.pop(name)
            elif count:
                attempt=OUT/'retry_jobs'/name/str(count)
                result=attempt/'exit.json'
                # Restart only under this controller's lock; detect previous unfinished retry.
                if not result.exists():
                    running=attempt/'running.json'
                    if running.exists():
                        proc=Path('/proc')/str(read(running)['pid'])/'cmdline'
                        if proc.exists() and str(attempt).encode() in proc.read_bytes():continue
                    code=1
                else:code=read(result)['returncode']
            else:
                marker=JOBS/s['job']/'exit.json'
                if not marker.exists():continue
                code=read(marker)['returncode']
            if code==0:raise RuntimeError(f'{name} exited without its completion receipt')
            if count>=2:raise RuntimeError(f'{name} failed after two resumable retries')
            count+=1;retries[name]=count;save(state_path,retries)
            folder=OUT/'retry_jobs'/name/str(count);folder.mkdir(parents=True,exist_ok=True)
            log=(folder/'run.log').open('w');handles.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            command=[s['python'],s['script'],*s['args']]
            # Small supervisor records exit even if this controller disconnects.
            runner='import subprocess,json;from pathlib import Path;p=subprocess.run('+repr(command)+');Path('+repr(str(folder/'exit.json'))+').write_text(json.dumps(dict(returncode=p.returncode)));raise SystemExit(p.returncode)'
            child=subprocess.Popen([s['python'],'-c',runner],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            children[name]=child;save(folder/'running.json',dict(pid=child.pid,command=command))
        time.sleep(30)
    save(OUT/'STATUS.json',dict(state='scoring',at=time.time()))
    merge_and_score();reports()
    save(OUT/'STATUS.json',dict(state='complete',at=time.time(),emoca_clips=215,image_scored_clips=172))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--status-only',action='store_true');a=parser.parse_args()
    if a.status_only:print(json.dumps(progress()));raise SystemExit(0)
    with (OUT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:watch()
        except BaseException:
            save(OUT/'STATUS.json',dict(state='failed',at=time.time(),traceback=traceback.format_exc()));raise
