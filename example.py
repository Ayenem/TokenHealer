from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from token_healing import TokenBoundaryHealer

def generate(prompt, model, tokenizer):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    generation_config = GenerationConfig(
        temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=16,
        pad_token_id=model.config.pad_token_id,
    )
    output = model.generate(inputs=input_ids, generation_config=generation_config)
    return tokenizer.decode(output[0], skip_special_tokens=True)

model_name_or_path = 'TheBloke/deepseek-llm-7B-base-GPTQ'
completion_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map='auto',
    trust_remote_code=False,
    revision='main',
    use_cache=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

test_prompts = [
    'The link is <a href="http:',
    'what is the DNS address ?! ',
    'I read a book about ',
    'I read a book about',
    'An example ["like this"] and another example ['
]
prompt = test_prompts[0]
print(f'\nOriginal prompt:\n{prompt}\n')

output = generate(prompt, completion_model, tokenizer)
print(f'Generation with original prompt:\n{output}\n')

token_healer = TokenBoundaryHealer(completion_model, tokenizer)
healed_prompt = token_healer(prompt)
print(f'Healed prompt:\n{healed_prompt}\n')
healed_output = generate(healed_prompt, completion_model, tokenizer)
print(f'Generation with healed prompt:\n{healed_output}\n')
