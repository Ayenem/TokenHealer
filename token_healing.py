from itertools import takewhile
from transformers.generation import MaxLengthCriteria
from transformers.generation import PrefixConstrainedLogitsProcessor as AllowedToks
from torch import IntTensor
from pygtrie import CharTrie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.model, self.vocab_trie = model, CharTrie(tokenizer.get_vocab())
        self.encode, self.decode = tokenizer.encode, tokenizer.decode
        self.max_length_1, self.use_cache = model.config.use_cache, MaxLengthCriteria(1)

    def __call__(self, prompt: str) -> str:
        prompt_ids = self.encode(prompt, return_tensors='pt').cuda()
        if alts := self.get_tail_alts(prompt_ids):
            healed_ids = self.regen_toks(prompt_ids[:, : -len(alts) or None], alts)
            prompt = self.decode(healed_ids.squeeze(), skip_special_tokens=True)
        return prompt

    def get_tail_alts(self, prompt_ids: IntTensor) -> IntTensor:
        prompt_toks = [*map(self.decode, prompt_ids.squeeze())]

        tail_toks_extensions = ( # ids of e.g. ['.', ':'] -> [['.', '. '], [':', '://']]
            self.vocab_trie.values(prefix=tail_tok.lstrip()) for tail_tok in reversed(prompt_toks)
        ) # querying contiguous tail tokens for alternative tokens
        toks_alts = [*takewhile(lambda exts: len(exts) > 1, tail_toks_extensions)]

        return toks_alts

    def regen_toks(self, ids: IntTensor, toks_alts: list[list[int]]) -> IntTensor:
        past_kv = None
        for alts in reversed(toks_alts): # regenerate last trimmed toks first
            ids = self.model.greedy_search(
                ids,
                logits_processor=AllowedToks(lambda *_, toks=alts: toks, 1),
                stopping_criteria=self.max_length_1,
                pad_token_id=self.model.config.pad_token_id,
                return_dict_in_generate=self.use_cache,
                past_key_values=past_kv,
            )
            if self.use_cache:
                ids, past_kv = ids.sequences, ids.past_key_values #type: ignore
        return ids
