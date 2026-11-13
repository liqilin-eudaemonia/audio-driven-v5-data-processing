"""LM Format Enforcer public API bridge, independent of Transformers import layout."""
from functools import lru_cache


@lru_cache(maxsize=4)
def tokenizer_data(tokenizer):
    from lmformatenforcer import TokenEnforcerTokenizerData
    # Decode after a digit to preserve start-of-word whitespace. Exclude control
    # tokens from the character trie; the parser enables EOS only at completion.
    anchor = tokenizer.encode("0", add_special_tokens=False)[-1]
    special = set(tokenizer.all_special_ids)
    entries = []
    for token in range(len(tokenizer)):
        if token in special:
            continue
        text = tokenizer.decode([anchor, token], clean_up_tokenization_spaces=False)[1:]
        alone = tokenizer.decode([token], clean_up_tokenization_spaces=False)
        entries.append((token, text, len(text) > len(alone)))
    def decode(tokens):
        return tokenizer.decode(tokens, clean_up_tokenization_spaces=False).rstrip("\ufffd")
    return TokenEnforcerTokenizerData(entries, decode, tokenizer.eos_token_id, False, len(tokenizer))


def prefix_constraint(tokenizer, schema):
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    enforcer = TokenEnforcer(tokenizer_data(tokenizer), JsonSchemaParser(schema))
    def allowed(batch_id, tokens):
        return enforcer.get_allowed_tokens(tokens.tolist()).allowed_tokens
    return allowed
