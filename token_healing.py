from itertools import takewhile
from transformers.generation import PrefixConstrainedLogitsProcessor, MaxLengthCriteria
from pygtrie import CharTrie

class TokenBoundaryHealer:

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.vocab_trie = CharTrie(tokenizer.get_vocab())
        self.encode, self.decode = tokenizer.encode, tokenizer.decode
        self.batch_decode = tokenizer.batch_decode

    def __call__(self, prompt: str) -> str:
        trimmed_prompt, trimmed_toks_alts = self.trim_prompt(prompt)
        prompt_ids = self.encode(trimmed_prompt, return_tensors='pt').cuda()
        max_length_1 = MaxLengthCriteria(1)
        def logits_rule(f): return PrefixConstrainedLogitsProcessor(f, num_beams=1)
        for tok_alts in trimmed_toks_alts:
            prompt_ids = self.model.greedy_search(
                prompt_ids,
                logits_processor=logits_rule(lambda *_, allowed_toks=tok_alts: allowed_toks),
                stopping_criteria=max_length_1,
                pad_token_id=self.model.config.pad_token_id,
            )
        prompt = self.batch_decode(prompt_ids, skip_special_tokens=True)[0]
        return prompt

    def trim_prompt(self, prompt: str) -> tuple[str, list[list[int]]]:
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        prompt_toks = self.batch_decode(encoded['input_ids'])

        tail_toks_extensions = ( # ids of e.g. ['.', ':'] -> [['.', '. '], [':', '://']]
            self.vocab_trie.values(prefix=tail_tok.strip()) for tail_tok in reversed(prompt_toks)
        ) # temporary reversing to target tail tokens for finding alternative tokens
        trimmed_toks_alts = [*takewhile(lambda exts: len(exts) > 1, tail_toks_extensions)]
        trimmed_toks_alts.reverse()

        last_trimmed_pos = encoded['offset_mapping'][-len(trimmed_toks_alts)][0]
        return prompt[: last_trimmed_pos or None], trimmed_toks_alts
