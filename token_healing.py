from itertools import takewhile
from transformers.generation import PrefixConstrainedLogitsProcessor, MaxLengthCriteria
from torch import IntTensor
from pygtrie import CharTrie

def allowed_toks(toks): return PrefixConstrainedLogitsProcessor(lambda *_: toks, num_beams=1)

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.model, self.vocab_trie = model, CharTrie(tokenizer.get_vocab())
        self.encode, self.decode = tokenizer.encode, tokenizer.decode
        self.use_cache, self._gen_kwargs = model.config.use_cache, {
            'stopping_criteria': MaxLengthCriteria(1),
            'pad_token_id': model.config.pad_token_id,
        }

    def __call__(self, prompt: str) -> str:
        left_ids, toks_alts = self.trim_prompt(prompt)
        if not toks_alts: return prompt
        left_ids = self.regenerate_tokens(left_ids, toks_alts)
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

    def regenerate_tokens(self, ids: IntTensor, toks_alts: list[list[int]]) -> IntTensor:
        past_kv = None
        for tok_alts in reversed(toks_alts): # regenerate last trimmed toks first
            ids = self.model.greedy_search(
                ids,
                logits_processor=allowed_toks(tok_alts),
                return_dict_in_generate=self.use_cache,
                past_key_values=past_kv,
                **self._gen_kwargs,
            )
            if self.use_cache:
                ids, past_kv = ids.sequences, ids.past_key_values #type: ignore
        return ids
