"""Keep coefficients fixed; isolate released metric and dataset conventions."""
import contextlib
import csv
import io
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
RUN=ROOT/'experiments/p2_avatarforcing_table4_recheck_20261110'
sys.path[:0]=[str(ROOT/'compare/deployment'),str(ROOT/'scripts')]
from p2_avatarforcing_table4_generate import read,save,sha
from vico_literature_metrics import evaluate,parity


def main():
    policy=read(RUN/'protocol.json');ids=[s['id'] for s in policy['samples']]
    arrays={(uid,role):np.load(RUN/'features'/role/(uid+'.npz'))['motion']
            for uid in ids for role in ['gt','avatarforcing','speaker']}
    profiles={}
    def score(name,selected,offset=False):
        g,p,x=[[arrays[uid,role] for uid in selected] for role in ['gt','avatarforcing','speaker']]
        if offset:
            g,p=[a[1:] for a in g],[a[1:] for a in p]
            x=[a[:-1] for a in x]
        with contextlib.redirect_stdout(io.StringIO()):answer=evaluate(g,p,x)
        profiles[name]=dict(ids=selected,clips=len(g),frames=sum(map(len,g)),metrics=answer['metrics'])
        return g,p,x
    g,p,x=score('released_DIM_kernels_same43_same25fps_coefficients',ids)
    check=parity(g,p,x);save(WORK/'DIM_KERNEL_PARITY.json',check)
    score('released_DIM_kernels_same43_listener_plus1_speaker0_proxy',ids,True)
    metadata=list(csv.DictReader(Path('/public/0_DATA/2_Avatar/ViCo/dataset/metadata.csv').open()))
    test=[r for r in metadata if r['data_split']=='test'];assert {r['uuid'] for r in test}==set(ids)
    mapping={r['audio']:r for r in metadata}
    mapped=[mapping[r['audio']]['listener'] for r in test]
    assert all(uid in ids for uid in mapped)
    score('DIM_audio_key_aliases_with_repeated_loader_rows_proxy',mapped)
    frame30={s['id']:s['frames'] for s in policy['samples']}
    retained=[uid for uid in mapped if 5<=frame30[uid]<=1024]
    score('DIM_aliases_and_30fps_length_limit_proxy',retained)
    scope=dict(test_rows=len(test),unique_test_audio_keys=len(set(r['audio'] for r in test)),
        aliases=[dict(requested=r['uuid'],audio=r['audio'],mapped=uid) for r,uid in zip(test,mapped) if r['uuid']!=uid],
        rows_after_length_proxy=len(retained),unique_pairs_after_aliases=len(set(mapped)),
        unique_pairs_after_aliases_and_length_proxy=len(set(retained)),
        caveat='This is a source-code projection on available SPECTRE arrays, not the unpublished author HDF/pickle contents. Exact HuBERT lengths remain unavailable; 25fps coefficients are not converted into native30fps EMOCA.')
    result=dict(coefficients='unchanged SPECTRE56 from sealed43, no rescaling or new generation',
        diagnostic_only=True,not_a_Table4_reproduction=True,profiles=profiles,dataset_projection=scope,
        math_parity=check,implementation_sha256=sha(__file__),
        old_results_sha256=sha(RUN/'RESULTS.json'))
    save(WORK/'METRIC_CROSSCHECK.json',result)
    for name,p in profiles.items():
        print(name,p['clips'],{k:p['metrics'][k] for k in ['FD','SID','Var','rPCC']},flush=True)
    print('SCOPE',scope,flush=True)


if __name__=='__main__':main()
