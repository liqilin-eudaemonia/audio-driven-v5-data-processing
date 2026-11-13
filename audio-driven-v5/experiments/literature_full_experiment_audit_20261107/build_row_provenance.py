"""Per-method result provenance; never infer a checkpoint from public availability."""
import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
OWN = {'GLARE': 'GLARE', 'ViCo / RLHG': 'RLHG', 'Learning to Listen / L2L': 'L2L',
       'PCH / PCHG': 'PCH', 'MFR-Net': 'MFR-Net', 'ELP': 'ELP',
       'CustomListener': 'CustomListener', 'DIM': 'DIM', 'ReactFace': 'ReactFace',
       'DiffListener': 'DiffListener', 'INFP': 'INFP', 'ListenFormer': 'ListenFormer',
       'SpeakerVid-5M': 'SpeakerVid-5M', 'ARIG': 'ARIG', 'DiTaiListener': 'DiTaiListener',
       'Avatar Forcing': 'Avatar-Forcing', 'ECHO': 'ECHO', 'OmniResponse': 'OmniResponse',
       'DyStream': 'DyStream', 'OmniHuman-1.5': 'OmniHuman-1.5', 'OmniPersona': 'OmniPersona'}


def decide(paper, table, method, group):
    """Return a claim bounded by the local PDF/supplement statement, not repo state."""
    name, number = paper['name'], table['table']
    non_table = table.get('record_type') in ['qualitative', 'human_text']
    result = dict(classification='evaluation_reported_training_checkpoint_unreported',
                  result_source=None, metric_scopes=[],
                  explanation='原文报告比较，未逐行披露训练、checkpoint或明确引用来源；不因资源可取而补写官方权重推理。')

    def set_result(kind, explanation, source=None, scopes=None):
        result.update(classification=kind, explanation=explanation,
                      result_source=source, metric_scopes=scopes or [])

    if method == OWN[name] or method == 'DIM-Speaker':
        set_result('authors_own_model', '作者自身训练模型／配置；不推断外部下载checkpoint，具体训练设置见逐篇记录。')
    elif group == 'controls':
        set_result('authors_constructed_control', '作者按本文定义构造随机／检索／镜像／Trans-AE等控制；不需要独立公开预训练模型。')
    elif group == 'long_video_generation_strategies':
        set_result('authors_generation_strategy_control', '作者在生成模型上实现的长视频推理策略；不是独立模型checkpoint。')
    elif group == 'paper_internal_baselines' and name == 'OmniResponse':
        set_result('authors_new_baseline', 'OmniResponse作者自行构建LSTM或AV-LLM对照，不是声称使用已发布同名checkpoint。')

    if name == 'PCH / PCHG' and number == '1':
        set_result('official_challenge_leaderboard', '原文明确截取ViCo官方最终leaderboard；包括作者自身队伍与其他队伍，不是逐队重训。',
                   dict(paper='ViCo challenge leaderboard', table='talking/listening tracks', url='https://vico-challenge.github.io/'))
    elif name == 'GLARE' and method != 'GLARE':
        set_result('target_dataset_training_declared', 'p7明确全部方法分别在作者整理RealTalk/Seamless数据上训练和测试；未披露各baseline初始化或checkpoint。')
    elif name == 'Learning to Listen / L2L' and method == 'LFI':
        set_result('target_dataset_retraining_declared',
                   'LFI在L2L数据上按原配置重训；补充D正文的LFI原数据实验则是原split、motion-only重训；不能称LFI官方checkpoint推理。')
    elif name == 'Learning to Listen / L2L' and method == 'L2L-Random-Walk':
        set_result('authors_constructed_motion_control', '作者在自身训练的VQ motion空间构造random-walk控制，不是另外下载的独立L2L生成器。')
    elif name == 'ELP':
        if number == '1上半' and method == 'RLHG':
            set_result('mixed_metric_cell_provenance', '只在表情FD与head FD两格标*并称沿官方报告；其余格未给同样来源声明。',
                       scopes=[dict(metrics=['FD_expression', 'FD_head'], kind='explicit_official_report_quote',
                                    source=dict(paper='ViCo / RLHG', table='3')),
                               dict(metrics=['remaining_metrics'], kind='not_explicitly_disclosed')])
        elif number == '1下半' and method == 'L2L':
            set_result('mixed_metric_cell_provenance', 'FD/V-D/SI-D/RPCC的expression/head格标*；WTLCC/STS/blink未标*。作者引用声明不自动证明尺度／PCC变换相同。',
                       scopes=[dict(metrics=['FD', 'V-D', 'SI-D', 'RPCC'], components=['expression', 'head'],
                                    kind='explicit_official_report_quote', source=dict(paper='Learning to Listen / L2L', table='1')),
                               dict(metrics=['WTLCC', 'STS', 'blink_WTLCC'], kind='not_explicitly_disclosed')])
        elif method == 'PCH' and not non_table and number != '4':
            set_result('authors_reimplementation_and_retraining', 'PCH只有部分原代码；ELP明确基于RLHG重实现并在目标数据训练；L2L下半主表实际没有PCH行。')
        elif number == '1上半' and method == 'L2L' or number == '1下半' and method == 'RLHG':
            set_result('target_dataset_retraining_declared', 'p6明确在该目标数据重新训练；不推断原人物checkpoint或从零初始化。')
        elif non_table and method in ['RLHG', 'PCH']:
            set_result('project_demo_visual_reuse', 'p7明确与官网显示结果手动对齐；不是独立重跑该方法。',
                       dict(paper='ViCo / RLHG' if method == 'RLHG' else 'PCH / PCHG', table='project demonstration'))
        elif non_table and method == 'L2L':
            set_result('motion_origin_unreported_renderer_retrained', '作者另训person-specific Vid2Vid可视化L2L motion；没有原L2L写实renderer代码，不能归为官方renderer推理。')
    elif name == 'CustomListener':
        if method == 'MFR-Net':
            set_result('explicit_paper_result_quote', '†明确引用MFR-Net原文数值；不是作者重训。', dict(paper='MFR-Net', table='1/2'))
        elif method in ['RLHG', 'PCH', 'L2L']:
            set_result('target_dataset_retraining_declared', 'ViCo正文与RealTalk补充均明确重训；不能跨数据集移动数值。')
        elif method == 'ELP':
            set_result('project_demo_visual_reuse', 'ELP无代码且表示不同；取其展示的source speaker视频生成自身比较，未运行原ELP。', dict(paper='ELP', table='project demonstration'))
    elif name == 'MFR-Net' and method in ['RLHG', 'PCH']:
        set_result('target_dataset_training_declared', '实验设置明确每种方法在ViCo训练集训练并评价test/OOD；具体初始化／权重版本未披露。')
    elif name == 'DIM':
        if method == 'ELP':
            set_result('authors_reimplementation_and_retraining', '因ELP无代码而重新实现；统一EMOCA56训练；不是原ELP100维输出直接测评。')
        elif method == 'L2L-CANDOR':
            set_result('authors_extra_pretraining_and_target_training', 'DIM作者先在CANDOR预训练L2L再目标数据训练；与普通L2L行不同。')
        elif method in ['RLHG', 'L2L', 'MeshTalk', 'FaceFormer', 'CodeTalker']:
            set_result('target_dataset_retraining_declared', '使用原代码路径并在本文目标表示／数据重训；BiWi变成5023顶点，不能当原模型官方权重零样本。')
    elif name == 'ReactFace' and method in ['LFI', 'L2L', 'RLHG']:
        if number == '9':
            set_result('evaluation_reported_training_checkpoint_unreported', 'RLD附加实验遵循RLHG 3DMM并修改自身loss，但未逐行给这些baseline checkpoint/重训细则；不机械继承REACT主表声明。')
        else:
            set_result('authors_reimplementation_and_retraining', 'p12明确reproduce并遵循原论文训练设置；共享本文FaceVerse/PIRender适配。')
    elif name == 'DiffListener':
        if method in ['RLHG', 'PCH', 'L2L']:
            set_result('target_dataset_retraining_declared', '正文及补充A.3明确在ViCo/RealTalk重训；RLHG/PCH均去掉emotion标签。')
        elif method in ['MFR-Net', 'CustomListener']:
            set_result('explicit_paper_result_quote', '补充A.3因未开源而引用ViCo原论文结果；不是本次生成；CustomListener条件差异须注明。',
                       dict(paper=method, table='1/2'))
    elif name == 'INFP':
        if number == '3' and method in ['L2L', 'RLHG', 'DIM']:
            set_result('explicit_paper_result_quote', '正文明确继承DIM定量/定性结果；L2L对应普通行，不是CANDOR†。', dict(paper='DIM', table='1'))
        elif method == 'DIM' and number in ['1/2', 'Fig4']:
            set_result('target_dataset_retraining_declared', '用DIM官方代码在DyConv重训speaker/listener分支及renderer；不是ViCo原模型推理。')
        elif number == '官方补充Fig4' and method in ['L2L', 'RLHG', 'DIM']:
            set_result('explicit_visual_result_reuse', '正文明确listener比较继承DIM定性结果；自身INFP直接迁移。', dict(paper='DIM', table='listener visual results'))
    elif name == 'ListenFormer':
        if method == 'MFR-Net':
            set_result('explicit_paper_result_quote', '表注*明确沿用MFR-Net原报告。', dict(paper='MFR-Net', table='1/2'))
        elif method in ['RLHG', 'PCH', 'DSPN'] and number != '3':
            set_result('authors_system_reproduction_declared', 'ViCo表注及正文说明其余方法在作者系统复现；具体初始化/核心checkpoint未报告。')
    elif name == 'ARIG':
        if number == '2' and method in ['RLHG', 'L2L', 'DIM', 'INFP']:
            set_result('explicit_paper_result_quote', 'ViCo表注明确继承DIM与INFP。',
                       dict(paper='INFP' if method == 'INFP' else 'DIM', table='3' if method == 'INFP' else '1'))
        elif method == 'DIM' and number in ['1', '官方补充2']:
            set_result('target_dataset_retraining_declared', '在RealTalk以DIM官方代码重训，并用于额外人工研究。')
        elif method == 'INFP' and number == 'Fig5':
            set_result('project_demo_visual_reuse', '直接取INFP官网两条DyConv demo；缺真实user视频，ARIG输入user视频另由audio合成。', dict(paper='INFP', table='project demo'))
    elif name == 'DiTaiListener' and method == 'INFP':
        set_result('explicit_paper_result_quote', 'ViCo动作Table4的†说明原法未发布而引用INFP报告；不是本论文在RealTalk训练INFP。', dict(paper='INFP', table='3'))
    elif name == 'Avatar Forcing':
        if method == 'INFP-reimpl-AvatarForcing':
            set_result('authors_reimplementation', '附录明确INFP*在FLOAT motion latent空间重实现；不是INFP官方核心模型。')
        elif method in ['RLHG', 'L2L', 'DIM'] and number == '4':
            set_result('explicit_paper_result_quote', 'ViCo表†明确取DIM定量结果。', dict(paper='DIM', table='1'))
    elif name == 'ECHO':
        if method == 'INFP-reimpl-AvatarForcing':
            set_result('explicit_reimplementation_result_source', '表注明确*表示Avatar Forcing重实现的INFP，不能改称原INFP。', dict(paper='Avatar Forcing', table='1/3/4 (per dataset)'))
        elif number == 'Fig5' and method != 'ECHO':
            set_result('project_demo_visual_reuse', '图注明确这些比较方法未开源、沿公开visual comparison样例；没有独立定量行。', dict(paper=method, table='project demonstration'))
        elif method == 'Per-FLOAT':
            set_result('unresolved_raster_figure_label', 'Fig3原图确有此标签，但无实现定义，图内引用编号与当前参考文献不一致；不合并到其他方法。')
        elif number in ['1', '2', '3'] and method != 'ECHO':
            set_result('numerical_match_or_comparison_only_not_explicit_quote', '存在与先前论文相同数值／比较行，但原文未逐行声明引用还是重跑；只记线索，不提升证据级别。')
    elif name == 'DyStream' and method in ['INFP-reimpl-DyStream', 'INFP-I']:
        set_result('authors_reimplementation', 'sec4.3明确INFP为作者reproduced版本；INFP-I进一步改为inpainting策略；非原INFP。')
    elif name == 'OmniPersona':
        if number == '2' and method == 'SpeakerVid-5M':
            set_result('numerical_match_only_not_explicit_quote', '与SpeakerVid原文最终dyadic行数值一致；没有找到明确引用/重跑声明。', dict(paper='SpeakerVid-5M', table='2 (candidate source, unconfirmed)'))
        elif number == '3' and method != 'OmniPersona':
            set_result('numerical_match_only_not_explicit_quote', '与原OmniResponse表逐项相同；没有明确引用声明，不擅自判为官方checkpoint或作者重训。', dict(paper='OmniResponse', table='2 (candidate source, unconfirmed)'))
        elif number == '13' and method != 'OmniPersona':
            set_result('authors_control_experiment_checkpoint_unreported', '作者报告统一SpeakerVid生成response text控制；这不说明共享generator/训练或baseline checkpoint来源。')
        elif method == 'Qwen2.5-Omni+OVI':
            set_result('authors_cascade_baseline_checkpoint_details_partial', '作者构建Qwen2.5-Omni→OVI级联；组件公开不等于基线完整适配/提示已经公开。')

    # Human scoring is an independent acquisition step, not proof of training origin.
    human_keys = {('ViCo / RLHG', '6'), ('ELP', '4'), ('CustomListener', '4'),
                  ('DIM', '附录2'), ('ReactFace', '8'), ('ListenFormer', '4'),
                  ('DiTaiListener', '1'), ('OmniResponse', '4'),
                  ('DiffListener', '官方补充6'), ('ARIG', '官方补充2')}
    mixed_keys = {('MFR-Net', '2/3'), ('INFP', '1/2'), ('Avatar Forcing', '1/2'),
                  ('ECHO', '4'), ('OmniHuman-1.5', '1/2'), ('OmniHuman-1.5', '3'),
                  ('OmniHuman-1.5', '4右/Fig5'), ('OmniPersona', '4/5')}
    human = table.get('record_type') == 'human_text' or (name, number) in human_keys
    result['evaluation_acquisition'] = ('authors_automatic_and_human_evaluation' if (name, number) in mixed_keys else
                                        'authors_human_study' if human else
                                        'visual_demonstration' if non_table else
                                        'see_classification_and_metric_scopes')
    return result


def main():
    papers = json.loads((BASE / 'PAPER_CATALOG.json').read_text())
    registry = {r['label']: r for r in json.loads((BASE / 'METHOD_REGISTRY.json').read_text())}
    rows = []
    for paper in papers:
        for table in paper['tables'] + paper.get('comparisons_outside_tables', []):
            for method in table['methods']:
                row = dict(paper=paper['name'], source_pdf=paper['pdf'], location=table['table'],
                           record_type=table.get('record_type', 'table'), pages=table['pdf_pages'],
                           dataset=table['dataset'], method=method,
                           table_context_evidence=table['provenance'])
                row.update(decide(paper, table, method, registry[method]['group']))
                rows.append(row)
    (BASE / 'ROW_PROVENANCE.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    fields = ['paper', 'location', 'record_type', 'pages', 'dataset', 'method', 'classification',
              'result_source', 'metric_scopes', 'evaluation_acquisition', 'explanation',
              'source_pdf', 'table_context_evidence']
    with (BASE / 'ROW_PROVENANCE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)
    stats = Counter(r['classification'] for r in rows)
    (BASE / 'ROW_PROVENANCE_COUNTS.json').write_text(json.dumps(dict(rows=len(rows), classes=stats,
        assertion='One record per method/comparison occurrence; mixed ELP metrics use cell-scoped overrides; unreported is not silently resolved.'), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(rows=len(rows), classes=stats), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
