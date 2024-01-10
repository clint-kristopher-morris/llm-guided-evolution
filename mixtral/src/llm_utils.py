import re
import time
import numpy as np
import transformers
from torch import bfloat16

import huggingface_hub
from huggingface_hub import InferenceClient

# Function to load and split the file
def split_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Regular expression for the pattern
    pattern = r"# --OPTION--"
    parts = re.split(pattern, content)

    return parts


def submit_mixtral(txt2mixtral, max_new_tokens=764, top_p=0.15, temperature=0.1, model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", lite=True):
    max_new_tokens = np.random.randint(600, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    if lite:
        huggingface_hub.login(new_session=False)
        client = InferenceClient(model=model_id)
        client.headers["x-use-cache"] = "0"
        print(txt2mixtral)
        output_txt = client.text_generation(txt2mixtral, max_new_tokens=max_new_tokens, return_full_text=False, temperature=temperature, top_p=top_p, seed=101)

    else:    
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=bfloat16,
            device_map='auto'
        )
        model.eval()
        print(model.device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        generate_text = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            return_full_text=False,  # if using langchain set True
            task="text-generation",
            # we pass model parameters here too
            temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            top_p=top_p,  # select from top tokens whose probability add up to 15%
            top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
            max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
            repetition_penalty=1.1,  # if output begins repeating increase
            do_sample=True,
        )

        res = generate_text(txt2mixtral)
        output_txt = res[0]["generated_text"]
    print(output_txt)
    print("="*120);print(f'time to load in seconds: {round(time.time()-start_time)}');print("="*120)
    return output_txt