"""Independent current500h leave-one-component-out experiments."""
VERSION='p2-current500h-component-ablation-v1'
ARMS={'no_whisper':dict(whisper=False,feature=True,event=True),
      'no_feature':dict(whisper=True,feature=False,event=True),
      'no_event':dict(whisper=True,feature=True,event=False)}
