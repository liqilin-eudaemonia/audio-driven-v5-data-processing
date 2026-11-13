"""One long-lived metric engine across ready methods/blocks, no per-block model reload."""
from runtime import *
from scoring import score,destination
from report import RGB

def main():
    from resident_scorer_check import check
    from contextlib import redirect_stdout,redirect_stderr
    path=WORK/'checks';path.mkdir(exist_ok=True)
    with (path/f'scorer_check_{os.getpid()}.log').open('w') as log:
        with redirect_stdout(log),redirect_stderr(log):check()
    while True:
        resident_checkpoint()
        pending=False;processed=False
        for method in RGB:
            for i,ss in enumerate(groups()):
                if (destination(method,i)/'SUCCESS.json').exists():continue
                pending=True
                if not all((folder(method,s)/'complete.json').exists() for s in ss):continue
                score(method,i,in_process=True);processed=True
        if not pending:return
        if not processed:time.sleep(5)
if __name__=='__main__':main()
