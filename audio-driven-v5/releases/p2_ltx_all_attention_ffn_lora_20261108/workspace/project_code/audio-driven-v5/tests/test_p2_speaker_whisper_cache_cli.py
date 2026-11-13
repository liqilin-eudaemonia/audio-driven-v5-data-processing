"""Cache-only entry cannot accidentally select a training command or listener PCM."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('speaker_whisper_cache_cli', ROOT/'scripts/p2_j_speaker_whisper_cache.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_four_node_cache_commands_use_speaker_entry_and_full_global_ranks():
    cluster = cli.read(cli.CLUSTER)
    rows = cli.cache_commands(cluster, cli.DEFAULT, batch_size=32, read_workers=3)
    assert len(rows) == 4
    for rank,row in enumerate(rows):
        assert '--nnodes=4' in row and '--nproc_per_node=8' in row
        assert '--node_rank='+str(rank) in row and '--master_port=29844' in row
        assert str(ROOT/'scripts/p2_j_speaker_whisper_cache.py') in row
        assert 'extract' in row and not {'train','launch'} & set(row)
        assert row[row.index('--batch-size')+1] == '32'
        assert row[row.index('--read-workers')+1] == '3'


@pytest.mark.parametrize('key,value',[('batch_size',0),('cpu_threads',False),('master_port',65536)])
def test_cache_commands_reject_invalid_parameters(key,value):
    with pytest.raises(ValueError):
        cli.cache_commands(cli.read(cli.CLUSTER),cli.DEFAULT,**{key:value})


def test_cache_entry_rejects_training_command_before_loading_models():
    result = subprocess.run([sys.executable,str(ROOT/'scripts/p2_j_speaker_whisper_cache.py'),'train'],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode == 2 and 'invalid choice' in result.stderr
