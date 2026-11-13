"""Classify every comparison label without equating it with a runnable model."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
GROUPS = {
    'visual_listener': 'RLHG|PCH|L2L|MFR-Net|ELP|CustomListener|DIM|ReactFace|DiffListener|ListenFormer|DiTaiListener|GLARE|DSPN|LFI',
    'interactive_video_known_audio': 'INFP|ARIG|Avatar-Forcing|ECHO|DyStream|DualTalk|StreamAvatar',
    'autonomous_av_dialogue': 'SpeakerVid-5M|OmniResponse|OmniPersona',
    'audio_driven_animation_background': 'OmniHuman-1|OmniHuman-1.5|InterActHuman|Loopy|SkyReels-A1|FantasyTalking|OmniAvatar|MultiTalk|LiveAvatar|SadTalker|AniTalker|EchoMimic|Hallo|Hallo3|FLOAT|Sonic|Real3DPortrait',
    '3d_speech_animation': 'MeshTalk|FaceFormer|CodeTalker|DIM-Speaker',
    'audio_only': 'Moshi|F5-TTS',
    'language_only': 'GPT-4o|GPT-4|GPT-o1|Qwen-7B-Chat|Claude-Sonnet-4|Gemini-2.5-Flash|DeepSeek-R1',
    'cascade': 'Qwen2.5-Omni+OVI',
    'joint_av_backbone_background': 'OVI',
    'paper_internal_baselines': 'OmniResponse-LSTM|OmniResponse-AV-LLM',
    'reimplementations_and_variants': 'INFP-reimpl-AvatarForcing|INFP-reimpl-DyStream|INFP-I|L2L-CANDOR|L2L-Random-Walk',
    'unresolved_figure_variant': 'Per-FLOAT',
    'controls': 'Random|Simulation|Simulation-attitude|NN-motion|NN-audio|Median|Mirror|Delayed-Mirror|Random-Expression|DLS-Random|Trans-AE',
    'long_video_generation_strategies': 'Prompt-Travel|Teacher-Forcing',
    'challenge_teams': 'sysu_hcp|iLearn|Avatar-team|Digital_Human|THU-Talking|cheese|en_train|LIMMC',
}


def main():
    source = json.loads((BASE / 'METHOD_OCCURRENCES.json').read_text())
    labels = [(m, group) for group, text in GROUPS.items() for m in text.split('|')]
    assert len(labels) == len({m for m, _ in labels})
    lookup = dict(labels)
    assert set(source) == set(lookup), (set(source) - set(lookup), set(lookup) - set(source))
    rows = [dict(label=m, group=lookup[m], occurrences=occs,
                 resource_status='pending_per_method_resource_audit') for m, occs in source.items()]
    resource_path = BASE / 'RESOURCE_MATRIX.json'
    if resource_path.exists():
        resources = {r['label']: r for r in json.loads(resource_path.read_text())}
        assert set(resources) <= set(source), 'Resource labels must belong to the current comparison union'
        for row in rows:
            row.update({k: v for k, v in resources.get(row['label'], {}).items()
                        if k not in ['label', 'group', 'occurrences']})
    (BASE / 'METHOD_REGISTRY.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    counts = {g: len(s.split('|')) for g, s in GROUPS.items()}
    table_groups = sum(len(p['tables']) for p in json.loads((BASE / 'PAPER_CATALOG.json').read_text()))
    non_table_groups = sum(len(p.get('comparisons_outside_tables', [])) for p in json.loads((BASE / 'PAPER_CATALOG.json').read_text()))
    (BASE / 'METHOD_COUNTS.json').write_text(json.dumps(dict(
        scope=f'labels from {table_groups} comparative table groups and {non_table_groups} figure/body comparison groups in supplied PDFs and listed acquired textual supplements',
        raw_labels=len(rows), groups=counts,
        table_comparison_groups=table_groups, non_table_comparison_groups=non_table_groups,
        primary_listener_interactive_av_candidates=sum(counts[g] for g in
            ['visual_listener', 'interactive_video_known_audio', 'autonomous_av_dialogue']),
        resource_decisions=sum(r['resource_status'] != 'pending_per_method_resource_audit' for r in rows),
        no_claim_all_are_runnable=True), ensure_ascii=False, indent=2) + '\n')
    print(counts)


if __name__ == '__main__':
    main()
