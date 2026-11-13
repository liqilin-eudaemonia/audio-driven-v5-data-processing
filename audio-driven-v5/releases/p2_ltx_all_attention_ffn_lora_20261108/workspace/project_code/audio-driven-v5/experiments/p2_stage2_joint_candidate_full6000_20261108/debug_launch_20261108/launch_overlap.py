"""Explicit user-authorized overlap: preserve vLLM; debug stops after pristine step0."""
from pathlib import Path
import json,os,sys,time
ROOT=Path('/private/liqi_workspace/project_code/audio-driven-v5');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from p2_listen.contracts import read,atomic_json
from p2_stage2_joint.training import precheck
import p2_stage2_joint_train as cli
from listen_cluster import launch
out=ROOT/'experiments/p2_stage2_joint_candidate_full6000_20261108';logs=out/'debug_launch_20261108';c=read(out/'config.json');precheck(c);cluster=read(out/'cluster.json')
resources=cli.launcher().selected_resources(cluster)
for n,node in zip(cluster['nodes'],resources['nodes']):
 if n['name']!=node['node']:raise ValueError('Resource node routing mismatch')
 selected=[g for g in node['gpus']if g['gpu'] in n['gpus']]
 if len(selected)!=len(n['gpus']):raise ValueError('Incomplete GPU resources')
 if n['name']!='node11'and not node['selected_idle']:raise ValueError('Other selected GPUs became busy')
 if n['name']=='node11'and any(g['used_mib']>31000 or g['total_mib']-g['used_mib']<110000 for g in selected):raise ValueError('Insufficient memory for authorized overlap')
commands=cli.commands(c)
# Last env assignment wins; native node launcher passes it to every child rank.
commands[-1][1:1]=['P2_TRAIN_MEMORY_GIB=108']
# Existing network assignments occur later, so replace the existing assignment.
for i,s in enumerate(commands[-1]):
 if s.startswith('P2_TRAIN_MEMORY_GIB='):commands[-1][i]='P2_TRAIN_MEMORY_GIB=108'
atomic_json(logs/'OVERLAP_AUTHORIZED.json',dict(at=time.time(),scope='User explicitly: preserve vLLM and overlap four-machine debug',resources=resources,commands=commands,node11_training_memory_gib=108,other_nodes_memory_gib=134,reserved_gpu='node11:7',automatic_formal_training=False))
launch(cluster,commands,logs/'overlap_nodes')
atomic_json(logs/'DEBUG_WORKERS_EXITED.json',dict(at=time.time(),success=True,status=read(out/'status.json')))
