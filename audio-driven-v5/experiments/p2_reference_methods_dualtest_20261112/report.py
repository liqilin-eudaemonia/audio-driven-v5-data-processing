from runtime import *
import csv
METHODS=['dystream','avatarforcing','listenformer','l2l_conan','l2l_fallon','l2l_stephen','l2l_trevor','dualtalk']
LABELS=dict(dystream='DyStream (official)',avatarforcing='Avatar Forcing (official, audio-only)',listenformer='ListenFormer (local ViCo retraining)',dualtalk='DualTalk (native motion only)',**{'l2l_'+n:'L2L '+n+' + GPAvatar' for n in ['conan','fallon','stephen','trevor']})

def report():
    W=WORK;records=[]
    text='# 五方法：固定ResponseNet200＋Seamless200结果\n\n仅覆盖用户确认的历史固定双200；没有运行完整586/6624。L2L四套权重分别报告，同一方法不按GT选优。项目统一视频八项和原生DECA动作诊断分别列出；这些方法均未生成listener音轨，音频指标N/A。\n\n'
    if (W/'status.json').exists():
        import datetime
        status=read(W/'status.json');stamp=datetime.datetime.fromtimestamp(status['at'],datetime.timezone.utc).isoformat()
        text+='进度更新时间：'+stamp+'；状态：'+status['stage']+'。以下进度是产物数量，不代表完整200条评分已经汇总。\n\n| 方法 | 已生成视频 / 400 | 已生成原生动作 / 400 |\n|---|---:|---:|\n'
        for method in METHODS:
            text+='| '+LABELS[method]+' | '+str(status['video'].get(method,'N/A'))+' | '+str(status['motion'].get(method,'N/A'))+' |\n'
        text+='\n'
    for dataset in ['responsenet','seamless']:
        text+='## '+dataset+'固定200\n\n| 方法 | 状态 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | Emo-Acc ↑ | Emo-Div (生成/GT) | ΔResp ↓ |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for method in METHODS:
            path=W/'results'/f'{method}_{dataset}.json'
            if not path.exists():
                state='N/A：无已验证RGB映射' if method=='dualtalk' else '进行中'
                text+=f'| {LABELS[method]} | {state} | — | — | — | — | — | — | — | — |\n';continue
            r=read(path);m=r['metrics'];v=[m.get(k) for k in ['psnr','ssim','lpips','fid','fvd','emo_acc']]
            fmt=lambda x:'N/A' if x is None else f'{x:.6f}'
            text+='| '+LABELS[method]+' | '+r.get('status','complete')+' | '+' | '.join(map(fmt,v))+' | '+fmt(m.get('emo_div_generated'))+' / '+fmt(m.get('emo_div_gt'))+' | '+fmt(m.get('delta_resp'))+' |\n'
            records.append(dict(dataset=dataset,method=method,status=r.get('status'),requested_pairs=r['requested_pairs'],**{k:m.get(k) for k in ['psnr','ssim','lpips','fid','fvd','emo_acc','emo_div_generated','emo_div_gt','delta_resp']},source=str(path),source_sha256=sha(path)))
        text+='\nEmo-Acc使用0–1比例。partial表示有原评分器N/A/支持不足，具体覆盖见JSON，不按0补分。DyStream行复用已核验历史固定200成绩，其余行为本轮新推理评分。\n\n'
    text+='## 原生动作诊断\n\nDualTalk及L2L的DECA原生/代理动作结果与RGB八项不同，不能合并排名。DualTalk neck/global head对应仍未验证，不称论文完整协议复现。\n\n| 数据集 | 方法 | FD Exp | FD Pose | P-FD Exp | P-FD Pose | MSE Exp | MSE Pose |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    native=[]
    for dataset in ['responsenet','seamless']:
        for method in ['dualtalk','l2l_conan','l2l_fallon','l2l_stephen','l2l_trevor']:
            path=W/'results'/f'{method}_{dataset}_native.json'
            if not path.exists():continue
            r=read(path);m=r['metrics'];values=[m[k][g] for k in ['FD','P_FD','MSE'] for g in ['exp','pose']]
            text+='| '+dataset+' | '+method+' | '+' | '.join(f'{v:.6f}' for v in values)+' |\n'
            native.append(dict(dataset=dataset,method=method,**{k+'_'+g:m[k][g] for k in ['FD','P_FD','MSE','SID','Var','rPCC'] for g in ['exp','pose']}))
    text+='\n[运行与恢复](README.md) · [实时状态](status.json) · [图像CSV](image_metrics.csv)。有完成清单COMPLETE.json后才代表本轮全部结束。\n'
    (W/'RESULTS.md').write_text(text)
    for name,values,cols in [('image_metrics.csv',records,['dataset','method','status','requested_pairs','psnr','ssim','lpips','fid','fvd','emo_acc','emo_div_generated','emo_div_gt','delta_resp','source','source_sha256']),('native_motion_metrics.csv',native,['dataset','method',*[k+'_'+g for k in ['FD','P_FD','MSE','SID','Var','rPCC'] for g in ['exp','pose']]])]:
        with (W/name).open('w') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(values)
if __name__=='__main__':report()
