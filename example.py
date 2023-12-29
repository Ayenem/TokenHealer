from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from token_healing import TokenBoundaryHealer

def generate(query, model, tokenizer):
    prompt_template="""<|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {query}<|im_end|>
    <|im_start|>assistant
    """
    system_message = "You are a helpful assistant."

    prompt = prompt_template.format(system_message=system_message, query=query)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    generation_config = GenerationConfig(
        temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512,
        pad_token_id=model.config.pad_token_id
    )
    output = model.generate(inputs=input_ids, generation_config=generation_config)
    return tokenizer.decode(output[0])

model_name_or_path = "TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    trust_remote_code=False,
    revision="main"
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# test queries from
# https://github.com/guidance-ai/guidance/blob/5f7fa7f6eef6455e6940fe743c5bfdb557330d0b/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb
test_queries = [
    "The link is <a href=\"http:",
    "I read a book about ",
    "I read a book about",
    "An example [\"like this\"] and another example ["
]
query = test_queries[0]
print(f"\nQUERY:\n{query}\n")

# unguided_output = generate(query, model, tokenizer, heal_prompt=False)
# print(f"UNGUIDED:\n{unguided_output}\n\n")

token_healer = TokenBoundaryHealer(model, tokenizer)
query = token_healer(query)
print(f"\nHEALED QUERY:\n{query}\n")
# guided_output = generate(query, model, tokenizer)
# print(f"GUIDED:\n{guided_output}\n\n")
