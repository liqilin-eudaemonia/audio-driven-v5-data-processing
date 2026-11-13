"""Readable duration-weighted statistics; never applies visual admission."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_corpora/seamless_listen_local_full'


def main():
    read=lambda name:json.loads((OUT/name).read_text())
    summary=read('screening_summary.json');face=read('face_size_distribution.json');crop=read('crop_size_distribution.json')
    assert summary['utc']==face['utc']==crop['utc'],'Statistics snapshot is being replaced; retry after reporter finishes'
    lines=['# Seamless 切分、音质与尺寸分布', '',f"统计时间：{summary['utc']}。完成 {summary['complete_pairs']:,}/{summary['total_pairs']:,} 对；错误 {summary['error_pairs']} 对；剩余 {summary['pending_pairs']} 对。最终快照：{summary['final']}。",'',
        '本版不做切镜检测，不按人脸或裁框尺寸筛选。音质通过仍是候选，未完成最终裁剪或训练准入。train 主池、起始反应储备池、dev/test 保持隔离。', '',
        '| 分组 | 片段数 | 小时 |','|---|---:|---:|']
    for key in sorted(summary['hours']):lines.append(f"| {key} | {summary['counts'][key]:,} | {summary['hours'][key]:.3f} |")
    key='train/primary/keep_candidate'
    if key in face['pools']:
        pool=face['pools'][key];f=pool['roles']['listener'];c=crop['pools'][key]['roles']['listener'];seconds=pool['seconds']
        assert abs(sum(c['side_seconds'])+c['unmeasured_seconds']-seconds)<1e-5
        lines+=['','## 训练主池的 listener 尺寸','',f"音质主池 {seconds/3600:.3f} 小时；原生人脸未测量 {f['unmeasured_seconds']/3600:.3f} 小时；参考裁框未测量 {c['unmeasured_seconds']/3600:.3f} 小时。参考框中预计头部放不下 {c['estimated_head_not_fitting_seconds']/3600:.3f} 小时，仅标记、未排除。",'',
            '以下比例以全部音质主池时长为分母。原生人脸是关键点范围；参考裁框是向肩膀扩展的固定方框，两者不是同一个尺度。', '',
            '| 原生人脸短边 p10 至少 | 覆盖小时 | 占主池时长 |','|---|---:|---:|']
        for threshold,value in sorted(f['p10_short_side_threshold_seconds'].items(),key=lambda x:float(x[0])):
            lines.append(f"| {threshold}px | {value/3600:.3f} | {100*value/seconds:.2f}% |")
        lines+=['','p10 仅基于可测量帧，不代表全部帧合格；必须结合有效帧比例阅读。','',
            '| 原生参考裁框边长至少 | 覆盖小时 | 占主池时长 |','|---|---:|---:|']
        edges=crop['pixel_edges']
        for i,e in enumerate(edges[:-1]):
            if e==0:continue
            duration=sum(c['side_seconds'][i:]);lines.append(f"| {e}px | {duration/3600:.3f} | {100*duration/seconds:.2f}% |")
        lines+=['','上述阈值仅供比较，未用于筛选。参考框可能仍有头部适配、有效帧或构图问题；大框不等于脸足够大。', '',
            '| 参考框缩放至 512 后的人脸高度 | 小时 | 占主池时长 |','|---|---:|---:|']
        for i,value in enumerate(c['output_face_height_seconds']):
            if value:
                upper=edges[i+1];label=f"[{edges[i]}, {upper if upper is not None else '∞'})px"
                lines.append(f"| {label} | {value/3600:.3f} | {100*value/seconds:.2f}% |")
        lines+=['','| 框内面部面积占比 | 小时 | 占主池时长 |','|---|---:|---:|']
        for i,value in enumerate(c['face_area_in_crop_seconds']):
            if value:lines.append(f"| [{crop['area_fraction_edges'][i]:.0%}, {crop['area_fraction_edges'][i+1]:.0%}) | {value/3600:.3f} | {100*value/seconds:.2f}% |")
    (OUT/'size_distribution_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(utc=summary['utc'],complete=summary['complete_pairs'],final=summary['final'],report=str(OUT/'size_distribution_report.md'))))


if __name__=='__main__':main()
