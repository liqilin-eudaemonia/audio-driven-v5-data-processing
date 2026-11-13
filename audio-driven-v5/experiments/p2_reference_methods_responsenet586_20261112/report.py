from runtime import *
import csv,datetime
METHODS=['dystream','avatarforcing','listenformer','l2l_conan','l2l_fallon','l2l_stephen','l2l_trevor','dualtalk']
RGB=METHODS[:-1]
NATIVE=['dualtalk','l2l_conan','l2l_fallon','l2l_stephen','l2l_trevor']
LABELS={'dystream':'DyStream 官方','avatarforcing':'Avatar Forcing 官方，audio-only','listenformer':'ListenFormer 本地ViCo重训','dualtalk':'DualTalk 原生动作',**{'l2l_'+n:'L2L '+n+'＋GPAvatar' for n in ['conan','fallon','stephen','trevor']}}

def report():
    s=read(WORK/'status.json');text='# ResponseNet 完整test586：五方法结果\n\n本页仅统计完整586条；历史双200结果独立保留。L2L四套权重分别报告。ListenFormer使用本地ViCo重训；L2L公开GPAvatar渲染；DualTalk尚无已验证RGB映射。音频指标N/A。\n\n'
    text+='更新：'+datetime.datetime.fromtimestamp(s['at'],datetime.timezone.utc).isoformat()+'；状态：'+s['stage']+'。产物数量不代表全量分数已汇总。\n\n| 方法 | 视频 / 586 | 原生动作 / 586 | 视频N/A | 已评分请求 / 586 |\n|---|---:|---:|---:|---:|\n'
    for m in METHODS:text+='| '+LABELS[m]+' | '+str(s['video'].get(m,'N/A'))+' | '+str(s['motion'].get(m,'N/A'))+' | '+str(s['video_unavailable'].get(m,'N/A'))+' | '+str(s['scored'].get(m,'N/A'))+' |\n'
    text+='\n## 完整586视频八项\n\n| 方法 | 状态 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD ↓ | Emo-Acc ↑ | Emo-Div 生成/GT | ΔResp ↓ |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    rows=[];fmt=lambda x:'N/A' if x is None else f'{x:.6f}'
    for m in METHODS:
        p=WORK/'results'/f'{m}_responsenet.json'
        if not p.exists():
            state='N/A：无已验证RGB映射' if m=='dualtalk' else '进行中'
            text+='| '+LABELS[m]+' | '+state+' | — | — | — | — | — | — | — | — |\n';continue
        r=read(p);v=r['metrics'];text+='| '+LABELS[m]+' | '+r['status']+' | '+' | '.join(fmt(v.get(k)) for k in ['psnr','ssim','lpips','fid','fvd','emo_acc'])+' | '+fmt(v.get('emo_div_generated'))+' / '+fmt(v.get('emo_div_gt'))+' | '+fmt(v.get('delta_resp'))+' |\n'
        rows.append(dict(method=m,status=r['status'],requested=TOTAL,**v,source=str(p)))
    text+='\n完整FID/FVD按586条有效特征重新汇总。partial及N/A按原评分器支持情况保留，不补零。\n\n## DECA原生动作诊断\n\n这些动作诊断与统一RGB八项分开；DualTalk neck/global head代理映射仍有限制。\n\n| 方法 | FD Exp | FD Pose | P-FD Exp | P-FD Pose | MSE Exp | MSE Pose |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for m in NATIVE:
        p=WORK/'results'/f'{m}_responsenet_native.json'
        if p.exists():
            r=read(p);text+='| '+LABELS[m]+' | '+' | '.join(fmt(r['metrics'][k][g]) for k in ['FD','P_FD','MSE'] for g in ['exp','pose'])+' |\n'
    text+='\n[运行与Monitor说明](README.md) · [实时状态](status.json) · [GPU监控](MONITOR.json) · [CSV](image_metrics.csv)。只有根目录COMPLETE.json存在才表示全部586范围完成。\n'
    (WORK/'RESULTS.md').write_text(text)
    with (WORK/'image_metrics.csv').open('w') as f:
        names=['method','status','requested','psnr','ssim','lpips','fid','fvd','emo_acc','emo_div_generated','emo_div_gt','delta_resp','source'];writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)
