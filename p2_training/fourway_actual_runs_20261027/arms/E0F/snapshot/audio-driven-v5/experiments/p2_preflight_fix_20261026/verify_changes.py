"""Prove the release changes only startup I/O and honest source accounting."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
base=ROOT/'src/p2_listen';new=ROOT/'releases/p2_preflight_v1/src/p2_listen'
def dump(n):return ast.dump(n,include_attributes=False)
def functions(p):return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
a=functions(base/'training.py');b=functions(new/'training.py')
assert a.keys()==b.keys()
for name in a:
 if name!='train':assert dump(a[name])==dump(b[name]),name
original=a['train'];current=b['train']
start=next(i for i,n in enumerate(original.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='rows')
end=next(i for i,n in enumerate(original.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='formal_stats')
replacement=ast.parse("from .preflight import prepare_training\nprepared=prepare_training(c)\nrows,targets=prepared['rows'],prepared['targets']\nsignatures,canonical,stats=prepared['signatures'],prepared['canonical'],prepared['statistics']\n").body
original.body[start:end]=replacement
assert dump(original)==dump(current),'Changes outside train preflight block'
a=functions(base/'config.py');b=functions(new/'config.py');assert a.keys()==b.keys()
for name in a:
 if name!='source_identity':assert dump(a[name])==dump(b[name]),name
source=json.loads((ROOT/'experiments/p2_e1_train50h_lora_tb_20261026/training_release.json').read_text())['source_identity']
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source.items())
report=dict(status='passed',training_functions_except_preflight_ast_identical=True,train_body_only_preflight_changed=True,
    config_functions_except_source_accounting_ast_identical=True,original_E1_source_files_unchanged=len(source),
    overrides={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in new.glob('*.py')})
(OUT/'source_change_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('Source AST and original E1 source audit passed')
