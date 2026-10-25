from pathlib import Path
import json,subprocess,shlex,sys
B=Path(__file__).resolve().parents[2];t=json.loads((B/'pyannote_separation_20261024/remote_transport.json').read_text());c=json.loads(Path(t['config_path']).read_text());ssh=['ssh','-i',c['ssh_key'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','root@'+t['host']]
if __name__=='__main__':
 p=subprocess.run(ssh+[shlex.join(sys.argv[1:])]);sys.exit(p.returncode)
