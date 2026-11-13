"""Use the existing Gemma/AV processors with explicit, lossless text lengths."""


def sequence_length(tokens, maximum=4096):
    if maximum not in (1024, 2048, 4096) or type(tokens) is not int or tokens <= 0:
        raise ValueError('Invalid text length policy')
    if tokens > maximum:
        raise OverflowError(f'Guidance has {tokens} tokens, configured limit {maximum}; no truncation permitted')
    return next(n for n in (1024, 2048, 4096) if tokens <= n)


def encode_guidance(encoder, prompt, maximum):
    import torch
    wrapper = encoder.encoder.tokenizer
    # Match the existing wrapper's outer whitespace normalization; annotation
    # strings and the complete serialized prompt remain preserved on disk.
    ids = wrapper.tokenizer(prompt.strip(), truncation=False)['input_ids']
    length = sequence_length(len(ids), maximum)
    model_config = encoder.encoder.model.config.text_config
    if length > model_config.max_position_embeddings:
        raise ValueError('Requested context exceeds Gemma positional support')
    old_length = wrapper.max_length
    try:
        wrapper.max_length = length
        pairs = wrapper.tokenize_with_weights(prompt)['gemma']
        retained = [int(t) for t, valid in pairs if int(valid)]
        if retained != ids:
            raise ValueError('Tokenizer truncated or changed original token sequence')
        if len(pairs) != length:
            raise ValueError('Unexpected text padding length')
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            hidden, mask = encoder.encoder.encode(prompt)
            if int(mask.sum()) != len(ids):
                raise ValueError('Gemma did not receive all guidance tokens')
            result = encoder.processor.process_hidden_states(hidden, mask)
        positive = dict(video=result.video_encoding.detach(), audio=result.audio_encoding.detach(),
                        mask=result.attention_mask.bool(), prompt=prompt)
        if positive['video'].shape != (1,length,4096) or positive['audio'].shape != (1,length,2048):
            raise ValueError('Unexpected AV text context geometry')
        if not all(torch.isfinite(positive[k]).all() for k in ('video','audio')):
            raise ValueError('Nonfinite Gemma context')
        return positive, dict(tokens=len(ids), sequence_length=length, truncated=False)
    finally:
        wrapper.max_length = old_length
