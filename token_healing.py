from itertools import takewhile
from transformers.generation import PrefixConstrainedLogitsProcessor, MaxLengthCriteria
from torch import IntTensor
from pygtrie import CharTrie

def allowed_toks(f): return PrefixConstrainedLogitsProcessor(f, num_beams=1)

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.model, self.vocab_trie = model, CharTrie(tokenizer.get_vocab())
        self.encode, self.decode = tokenizer.encode, tokenizer.decode
        self._gen_kwargs = {
            'stopping_criteria': MaxLengthCriteria(1),
            'pad_token_id': model.config.pad_token_id,
        }

    def __call__(self, prompt: str) -> str:
        left_ids, toks_alts = self.trim_prompt(prompt)
        if not toks_alts: return prompt
        if self.model.config.use_cache:
            past_key_values = None
            for tok_alts in reversed(toks_alts): # regenerate last trimmed toks first
                left_ids = self.model.greedy_search(
                    left_ids,
                    logits_processor=allowed_toks(lambda *_, alts=tok_alts: alts),
                    past_key_values=past_key_values,
                    return_dict_in_generate=True,
                    **self._gen_kwargs,
                )
                left_ids, past_key_values = left_ids.sequences, left_ids.past_key_values
        else:
            for tok_alts in reversed(toks_alts):
                left_ids = self.model.greedy_search(
                    left_ids,
                    logits_processor=allowed_toks(lambda *_, alts=tok_alts: alts),
                    **self._gen_kwargs,
                )
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
