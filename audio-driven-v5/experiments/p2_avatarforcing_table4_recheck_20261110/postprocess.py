"""Finish the CPU audit and refresh existing indices after the queued GPU run."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def replace_paragraph(path, prefix, text):
    original = path.read_text()
    paragraphs = original.split('\n\n')
    matches = [i for i, p in enumerate(paragraphs) if p.startswith(prefix)]
    assert len(matches) == 1, (str(path), matches)
    paragraphs[matches[0]] = text
    temporary = path.with_name(path.name + '.avatar-table4.tmp')
    temporary.write_text('\n\n'.join(paragraphs))
    temporary.replace(path)


def main():
    with (WORK / 'postprocess.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(WORK / 'POSTPROCESS_ACTIVE.json', dict(pid=os.getpid(), stage='waiting_controller', gpu=False))
        while not (WORK / 'CONTROLLER_COMPLETE.json').exists():
            status = read(WORK / 'status.json')
            if status.get('stage') == 'failed':
                save(WORK / 'POSTPROCESS_FAILED.json', dict(reason='upstream failed', status=status))
                raise RuntimeError(str(status))
            time.sleep(30)
        assert read(WORK / 'CONTROLLER_COMPLETE.json')['passed_execution']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        save(WORK / 'POSTPROCESS_ACTIVE.json', dict(pid=os.getpid(), stage='cpu_final_audit', gpu=False))
        with (WORK / 'final_audit.log').open('ab') as output:
            subprocess.run([str(ROOT / 'compare/.envs/vico_more/bin/python'), str(WORK / 'final_audit.py')],
                           env=env, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
        assert read(WORK / 'FINAL_AUDIT.json')['passed']
        result = read(WORK / 'RESULTS.json')
        matched = result['matched_cells_at_published_precision']
        finding = (f'8项中{matched}项按论文两位小数一致；'
                   + ('支持Avatar Forcing／SPECTRE这一分支数值一致，未证明作者私有运行配置完全相同。'
                      if matched == 8 else '当前公开配置重建未复现Table4，不能宣称评测已严格对齐。'))
        root_text = ('**Avatar Forcing Table4端到端检查已完成（2026-11-10）：** '
                     '官方权重＋真实speaker视觉，ViCo43全部生成，129路SPECTRE动作提取及FD／rPCC／SID／Var的Exp／Pose共8项评分完成；'
                     + finding + '完整视频解码、特征覆盖、独立统计重算和来源校验通过。'
                     '此分支采用SID K15／9，与DIM的EMOCA／K40／20分支不同；旧DIM best10保持暂停。'
                     '[结果、配置和核验](experiments/p2_avatarforcing_table4_recheck_20261110/RESULTS.md)。')
        eval_text = ('**Avatar Forcing Table4独立复现检查已完成（2026-11-10）：** '
                     '官方权重恢复真实speaker视觉，43条生成＋129路SPECTRE动作＋8项评分及独立最终核验完成。'
                     + finding + 'SID K15／9、逐系数Var；不等同DIM的EMOCA／K40／20、P-FD或MSE验证。'
                     '[逐项实测与论文差值](../experiments/p2_avatarforcing_table4_recheck_20261110/RESULTS.md) · '
                     '[冻结协议与来源](../experiments/p2_avatarforcing_table4_recheck_20261110/README.md)。')
        replace_paragraph(ROOT / 'README.md', '**Avatar Forcing Table4端到端检查', root_text)
        replace_paragraph(ROOT / 'evaluation/README.md', '**Avatar Forcing Table4独立复现检查', eval_text)
        local_text = ('**实际执行已完成。** 官方test43全部生成，129路SPECTRE动作与8项评分完成；'
                      + finding + '\n\n[逐项结果与论文差值](RESULTS.md) · [原始数值和敏感性结果](RESULTS.json) · '
                      '[最终独立核验](FINAL_AUDIT.json) · [完成回执](COMPLETE.json)。')
        replace_paragraph(WORK / 'README.md', '用户要求用已有官方权重推理', local_text)
        save(WORK / 'POSTPROCESS_COMPLETE.json', dict(passed=True, gpu=False, matched_cells=matched,
             final_audit=True, indices_updated=['README.md', 'evaluation/README.md'], at=time.time()))
        save(WORK / 'POSTPROCESS_ACTIVE.json', dict(pid=os.getpid(), stage='complete', gpu=False))
        print(f'Completed final audit and documentation; {matched}/8 matching cells', flush=True)


if __name__ == '__main__':
    main()
