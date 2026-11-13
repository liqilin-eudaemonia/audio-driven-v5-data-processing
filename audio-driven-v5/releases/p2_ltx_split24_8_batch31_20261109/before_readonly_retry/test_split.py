from pathlib import Path
import copy,hashlib,importlib,json,sys
import pytest
WORK=Path(__file__).resolve().parent;ROOT=WORK.parents[1]
sys.path[:0]=[str(WORK),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import control as C
C.activate()
import runtime as R
import torch
torch.set_num_threads(2)

@pytest.mark.parametrize('world,counts',[(24,[2]*7+[1]*17),(8,[4]*7+[3])])
def test_partition_is_exact_ordered_batch(world,counts):
    pairs=[(i*2,i*3)for i in range(31)]
    parts=[R.partition(pairs,r,world)for r in range(world)]
    assert [len(x)for x in parts]==counts
    assert [p for part in parts for p in part]==pairs
    for args in [(pairs,world,world),(pairs[:30],0,world),(pairs,0,31)]:
        with pytest.raises(ValueError):R.partition(*args)

@pytest.mark.parametrize('world',[8,24])
def test_host_and_native_local_sums_match_batch31_gradient_and_adam(world):
    from p2_ltx_events.heads import objective
    from p2_whisper_av.execution import accumulator
    c=dict(hidden_event_supervision=dict(clip_weight=1.,temporal_weight=0.),loss_weights=dict(av=1.,event=.05,video_feature=.025,audio_feature=.025))
    counts={'clip/nod':torch.tensor(16),'clip/vocal':torch.tensor(11)}
    def values(p,i):
        g=torch.Generator().manual_seed(42+i);x=torch.randn(2,generator=g);z=torch.randn(2,generator=g)
        return dict(av=(p*x-z).square().mean(),features=dict(video=(p*x).square().mean(),audio=(p-z).square().mean()),event_sums={'clip/nod':(p[0]-i/31).square()if i%2==0 else p.sum()*0,'clip/vocal':(p[1]+i/31).square()if i%3==0 else p.sum()*0})
    p=torch.nn.Parameter(torch.tensor([.2,-.3]));q=torch.nn.Parameter(p.detach().clone())
    execution=dict(gradient_reduction='streaming_fp64',gradient_bucket_mb=1)
    baseline=accumulator([p],31,execution)
    for i in range(31):objective(values(p,i),c,counts,31,1)[0].backward();baseline.add()
    baseline.finish(1)
    total=torch.zeros_like(q,dtype=torch.float64)
    for rank in range(world):
        part=R.partition(list(range(31)),rank,world)
        acc=R.HostFP64Accumulator([q],1)if world==8 else accumulator([q],len(part),execution)
        for i in part:objective(values(q,i),c,counts,31,1)[0].backward();acc.add()
        # Read exact local64 sum rather than round each local sum early.
        if hasattr(acc,'buffer'):total.add_(acc.buffer)
        else:total.add_(q.grad.double())
        q.grad=None
    q.grad=total.float();assert torch.equal(p.grad,q.grad)
    a=torch.optim.AdamW([p],lr=1e-4);b=torch.optim.AdamW([q],lr=1e-4);a.step();b.step()
    assert torch.equal(p,q)and C.equal_tree(a.state_dict(),b.state_dict())

def test_host_multibucket_dtype_and_unused_parameters():
    p=torch.nn.Parameter(torch.linspace(-1,1,150003));q=torch.nn.Parameter(torch.ones(7,dtype=torch.bfloat16));u=torch.nn.Parameter(torch.ones(2))
    expected=[torch.zeros_like(p,dtype=torch.float64),torch.zeros_like(q,dtype=torch.float64)]
    acc=R.HostFP64Accumulator([p,q,u],1)
    for i in range(3):
        p.grad=torch.sin(p.detach()+i);q.grad=torch.full_like(q,.1*(i+1))if i%2 else None
        expected[0]+=p.grad.double()
        if q.grad is not None:expected[1]+=q.grad.double()
        acc.add();assert p.grad is None and q.grad is None
    acc.finish();assert torch.equal(p.grad,expected[0].float())and torch.equal(q.grad,expected[1].bfloat16())and u.grad is None
    with pytest.raises(ValueError):acc.finish()

@pytest.mark.parametrize('arm,oldworld,world',[('full',31,8),('noevent',1,24)])
def test_native_model_adam_checkpoint_roundtrip(arm,oldworld,world,tmp_path):
    from p2_listen.training import rng_state
    fixture=importlib.import_module('test_p2_ltx_all_lora'+('_noevent'if arm=='noevent'else'')).fixture
    _,_,model,*_=fixture();optimizer=torch.optim.AdamW(model.groups(),weight_decay=0.)
    for p in model.parameters():
        if p.requires_grad:p.grad=torch.full_like(p,.01)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    identity=dict(world=oldworld,experiment=dict(output=str(old),training=dict(global_batch=31)),source={'native':'unchanged'})
    target=copy.deepcopy(identity);target.update(world=world,source=dict(identity['source'],split='hashed'));target['experiment']['output']=str(new)
    cp=C.native(arm,'checkpoint');source=old/'step_000007.pt';destination=new/source.name
    cp.save(source,model,optimizer,7,identity,[rng_state()for _ in range(oldworld)])
    digest=C.sha(source);proof=C.rewrite(arm,source,destination,target,identity)
    assert proof['model_and_adam_equal']and C.sha(source)==digest
    migrated=cp.checked(destination);assert len(migrated['rank_rng'])==world
    expected=copy.deepcopy(migrated['model']);state=copy.deepcopy(migrated['optimizer'])
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:p.add_(.1)
    cp.resume(destination,model,optimizer,target,world-1)
    assert C.equal_tree(expected,model.state_dict())and C.equal_tree(state,optimizer.state_dict())
    with pytest.raises(ValueError):C.rewrite(arm,source,source,target,identity)

@pytest.mark.parametrize('arm',['full','noevent'])
def test_generated_training_diff_is_explicit_and_bound(arm):
    spec=json.loads((WORK/'TRAINING_DIFF.json').read_text())[arm]
    original=Path(spec['native']).read_text();text=original
    for edit in spec['edits']:
        assert text.count(edit['old'])==edit['count'];text=text.replace(edit['old'],edit['new'])
    assert text==(WORK/f'training_{arm}.py').read_text()
    assert hashlib.sha256(original.encode()).hexdigest()==spec['native_sha256']
    module=R.load_training(arm)
    assert module.__package__==R.PACKAGES[arm]
    assert '/len(chosen)'not in text
    assert 'objective(values,c,counts,31,1)'in text
    assert module.source_identity()==R.source_identity(arm)

def test_report_uses_sample_weighting():
    values=list(range(31));expected=sum(values)/31
    for world in (8,24):
        result=sum(sum(R.partition(values,r,world))/31 for r in range(world))
        assert result==pytest.approx(expected)
