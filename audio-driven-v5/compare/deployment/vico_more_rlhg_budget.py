"""RLHG full-budget runner: measured delivery cutoff, unchanged native training math."""
import argparse
from vico_more_common import WORK,read,save,sha
import vico_more_train as native


def main():
    p=argparse.ArgumentParser();p.add_argument('--compile-loss',action='store_true');a=p.parse_args()
    budget=read(WORK/'rlhg/RUN_BUDGET.json')
    assert budget['eligible'] and budget['selected_compiled_loss']==a.compile_loss
    assert budget['delivery_unix']==1794110400 and budget['postprocessing_reserve_s']>=5400
    assert budget['model_freeze_unix']<=budget['delivery_unix']-5400
    assert budget['epochs']==500 and budget['gpu']==7
    native.FREEZE=budget['model_freeze_unix']
    save(WORK/'rlhg/EFFECTIVE_EXECUTION.json',dict(budget_sha256=sha(WORK/'rlhg/RUN_BUDGET.json'),
        wrapper_sha256=sha(__file__),native_training_sha256=sha(native.__file__),
        model_freeze_unix=native.FREEZE,epochs=500,compile_loss=a.compile_loss,
        numerical_change='none beyond the separately audited optional FP32 loss compiler'))
    native.train('rlhg',benchmark=False,compile_loss=a.compile_loss)


if __name__=='__main__':main()
