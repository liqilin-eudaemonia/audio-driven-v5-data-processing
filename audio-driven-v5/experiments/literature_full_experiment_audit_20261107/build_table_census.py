"""Numbered table coverage by role; text mentions are a cross-check, not reading proof."""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
# Every numbered table in each supplied unique research PDF is accounted for.
# Combined experiment-ledger groups may map to more than one physical table.
MAIN = {
 'GLARE': {'comparison': '2,3', 'own_ablation_or_metric_sensitivity': '4,8,9,10,11', 'dataset_statistics': '1,5', 'detector_or_training_settings': '6,7'},
 'ViCo / RLHG': {'comparison': '3,4,6', 'own_modality_architecture_ablation': '5,A1', 'dataset_statistics': '1,2'},
 'Learning to Listen / L2L': {'comparison': '1', 'own_ablation': '2'},
 'PCH / PCHG': {'challenge_leaderboard': '1', 'own_talking_task_ablation': '2,3'},
 'MFR-Net': {'comparison': '1,2,3', 'own_ablation': '4'},
 'ELP': {'comparison_or_control': '1,2,3,4'},
 'CustomListener': {'comparison': '1,2,4', 'own_ablation': '3'},
 'DIM': {'comparison': '1,2,4', 'own_ablation': '3'},
 'ReactFace': {'notation': '1', 'comparison': '2,8,9', 'own_component_or_loss_ablation': '3,4,5,6,7'},
 'DiffListener': {'comparison': '1,2', 'own_ablation': '3,4'},
 'INFP': {'comparison': '1,2,3,4'},
 'ListenFormer': {'comparison': '1,2,3,4', 'own_fusion_ablation': '5'},
 'SpeakerVid-5M': {'dataset_comparison': '1', 'own_baseline_settings': '2', 'annotation_prompts': '3,4'},
 'ARIG': {'comparison': '1,2,3'},
 'DiTaiListener': {'comparison': '1,2,3,4,5'},
 'Avatar Forcing': {'comparison': '1,2,3,4', 'own_ablation': '5,6,7'},
 'ECHO': {'comparison': '1,2,3,4'},
 'OmniResponse': {'dataset_comparison': '1', 'comparison': '2,4', 'own_ablation': '3'},
 'DyStream': {'comparison': '1,2,4', 'own_ablation': '3,5'},
 'OmniHuman-1.5': {'own_ablation': '1', 'mixed_own_ablation_and_comparison': '2', 'comparison': '3,4'},
 'OmniPersona': {'capability_claims_not_performance': '1', 'comparison': '2,3,4,5,13,15,16',
                 'architecture_training_conditions_or_rating_anchors': '6,7,8,9',
                 'own_probes_ablation_or_timing': '10,11,12,14,17'},
}

EXTRA = [
 dict(paper='DIM', document='17_DIM_ECCV2024.txt', phase='embedded_appendix',
      roles={'own_modality_ablation': ['1'], 'human_comparison': ['2']}, pages=[21, 22]),
 dict(paper='Learning to Listen / L2L', document='L2L_arxiv.txt', phase='official_arxiv_appendix',
      roles={'per_identity_comparison': ['3', '4', '5', '6'], 'TLCC_comparison': ['7']}, pages=[13, 14, 15, 16]),
 dict(paper='CustomListener', document='CustomListener_supp.txt', phase='official_CVF_supplement',
      roles={'RealTalk_comparison': ['5', '6']}, pages=[1, 2, 3, 4, 5]),
 dict(paper='DiffListener', document='DiffListener_supp_07114_supp.txt', phase='official_CVF_supplement',
      roles={'own_modality_ablation': ['5'], 'human_comparison': ['6']}, pages=[1, 2, 3, 4]),
 dict(paper='INFP', document='INFP_supp_9131_supplementary.txt', phase='official_CVF_supplement',
      roles={'own_data_size_memory_ablation': ['1']}, pages=[1, 2, 3]),
 dict(paper='ARIG', document='ARIG_CVF_supp.txt', phase='official_CVF_supplement',
      roles={'own_ablation': ['1'], 'human_comparison': ['2']}, pages=[1, 2, 3, 4, 5]),
 dict(paper='DiTaiListener', document='DiTaiListener_CVF_supp.txt', phase='official_CVF_supplement_newer_than_local',
      roles={'long_video_strategy_comparison': ['6'], 'own_CTM_ablation': ['7']}, pages=[1, 2, 3]),
]


def find_mentions(text):
    result = {}
    for page, content in enumerate(text.split('\f'), 1):
        for match in re.finditer(r'\b(?:Table|TABLE|Tab\.)\s*(A?\d+)\b', content):
            result.setdefault(match[1], set()).add(page)
    return {k: sorted(v) for k, v in result.items()}


def main():
    papers = json.loads((BASE / 'PAPER_CATALOG.json').read_text())
    assert set(MAIN) == {p['name'] for p in papers}
    rows = []
    for paper in papers:
        name = paper['name']
        roles = {role: ids.split(',') for role, ids in MAIN[name].items()}
        document = Path(paper['pdf']).stem + '.txt'
        mentions = find_mentions((BASE / 'texts' / document).read_text())
        annotated = {i for ids in roles.values() for i in ids}
        assert set(mentions) <= annotated, (name, set(mentions) - annotated)
        assert annotated <= set(mentions), (name, annotated - set(mentions))
        rows.append(dict(paper=name, document=document, phase='supplied_PDF', roles=roles,
                         table_mentions_pages=mentions,
                         distinct_numbered_tables=len(annotated),
                         caption_roles_curated=True))
    for item in EXTRA:
        entry = dict(item)
        entry['table_mentions_pages'] = find_mentions((BASE / 'texts' / item['document']).read_text())
        entry['distinct_numbered_tables'] = sum(map(len, item['roles'].values()))
        rows.append(entry)
    supplied_instances = sum(r['distinct_numbered_tables'] for r in rows[:len(papers)]) + 2
    external_instances = sum(r['distinct_numbered_tables'] for r in rows[len(papers):]
                             if r['phase'] != 'embedded_appendix')
    result = dict(scope='All numbered tables in supplied21 unique scientific PDFs, plus the listed acquired textual supplements',
                  supplied_numbered_table_instances=supplied_instances,
                  additional_external_table_instances=external_instances,
                  reviewed_numbered_table_instances=supplied_instances + external_instances,
                  entries=rows,
                  unresolved_or_nontextual_material=[
                      'ELP official ZIP contains video only, no PDF/HTML/text scientific supplement',
                      'OmniPersona AppendixA mentions unnamed concurrent-work comparison videos; no model identity can be inferred from unnamed material',
                      'ECHO Fig3 Per-FLOAT label and raster reference numbers conflict with current bibliography; retained as unresolved identity',
                      'OmniHuman1.5 anonymous CS-A/B/C/D and unnamed general Base Model are not mapped to guessed public models'],
                  scope_limit='Numbered-table closure and declared supplement review are not alone proof of every methodological/provenance inference.')
    (BASE / 'TABLE_CENSUS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 表格覆盖清单与排除理由', '',
             '本清单把数据／符号／训练设置／自身消融与跨方法性能比较分开，避免以“只查主结果表”作为无遗漏证明。图中和正文的其他比较另列于PAPER_DETAILS和ROW_PROVENANCE；同一作者方法的参数配置不各算一个独立模型。', '',
             '| 论文 | 主PDF内全部表号及角色 |', '|---|---|']
    for row in rows[:len(papers)]:
        detail = '；'.join(role + ': ' + '/'.join(ids) for role, ids in row['roles'].items())
        lines.append('| ' + row['paper'] + ' | ' + detail + ' |')
    lines += ['', '另外逐页检查的文字补充：', '']
    for row in rows[len(papers):]:
        lines.append('- ' + row['paper'] + ' / ' + row['phase'] + '：' +
                     '；'.join(role + ' ' + '/'.join(ids) for role, ids in row['roles'].items()) + '。')
    lines += ['', '保留的不可识别／非文字材料：', '']
    lines.extend('- ' + s for s in result['unresolved_or_nontextual_material'])
    lines += ['', 'JSON中的出现页包含正文引用页，不能当作表的唯一印刷页。DIM附录重用Table1/2编号，已按阶段另记；DiTai新版CVF补充比本地15页稿多Table6/7，未覆盖或篡改原PDF。', '']
    (BASE / 'TABLE_CENSUS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(papers=len(papers), supplied_numbered_table_instances=supplied_instances,
                         additional_external_table_instances=external_instances,
                         additional_supplement_records=len(EXTRA)), ensure_ascii=False))


if __name__ == '__main__':
    main()
