from itertools import takewhile
from transformers.generation import PrefixConstrainedLogitsProcessor, MaxLengthCriteria
from torch import IntTensor
from pygtrie import CharTrie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.model, self.use_cache = model, model.config.use_cache
        self.encode, self.decode = tokenizer.encode, tokenizer.decode
        self.vocab_trie = CharTrie(tokenizer.get_vocab())

    def __call__(self, prompt: str) -> str:
        left_ids, toks_alts = self.trim_prompt(prompt)
        if not toks_alts: return prompt
        max_length_1, past_kv = MaxLengthCriteria(1), None
        def allowed_toks(f): return PrefixConstrainedLogitsProcessor(f, num_beams=1)
        for tok_alts in reversed(toks_alts): # regenerate last trimmed toks first
            left_ids = self.model.greedy_search(
                left_ids,
                logits_processor=allowed_toks(lambda *_, alts=tok_alts: alts),
                stopping_criteria=max_length_1,
                pad_token_id=self.model.config.pad_token_id,
                return_dict_in_generate=self.use_cache,
                past_key_values=past_kv,
            )
            if self.use_cache:
                past_kv, left_ids = left_ids.past_key_values, left_ids.sequences

        healed_prompt = self.decode(left_ids.squeeze(), skip_special_tokens=True)
        return healed_prompt

    def trim_prompt(self, prompt: str) -> tuple[IntTensor, list[list[int]]]:
        prompt_ids = self.encode(prompt, return_tensors='pt').cuda()
        prompt_toks = [*map(self.decode, prompt_ids.squeeze())]

        tail_toks_extensions = ( # ids of e.g. ['.', ':'] -> [['.', '. '], [':', '://']]
            self.vocab_trie.values(prefix=tail_tok.lstrip()) for tail_tok in reversed(prompt_toks)
        ) # querying contiguous tail tokens for alternative tokens
        trimmed_toks_alts = [*takewhile(lambda exts: len(exts) > 1, tail_toks_extensions)]

        return prompt_ids[:, : -len(trimmed_toks_alts) or None], trimmed_toks_alts
