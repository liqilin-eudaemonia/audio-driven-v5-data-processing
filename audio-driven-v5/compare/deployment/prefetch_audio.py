from huggingface_hub import snapshot_download
print(snapshot_download('facebook/wav2vec2-base-960h',allow_patterns=['config.json','preprocessor_config.json','tokenizer_config.json','vocab.json','special_tokens_map.json','pytorch_model.bin']),flush=True)
