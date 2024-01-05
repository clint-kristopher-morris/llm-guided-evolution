import os
import re
from typing import Optional
import fire
from llama import Llama


def set_model_paths(model='7b'):
    model_paths = {'7b':{'ckpt_dir':"CodeLlama-7b/","tokenizer_path":"CodeLlama-7b/tokenizer.model"},
                   '13b':{'ckpt_dir':"CodeLlama-13b-Python/","tokenizer_path":"CodeLlama-13b-Python/tokenizer.model"},
                   '13I':{'ckpt_dir':"CodeLlama-13b-Instruct/","tokenizer_path":"CodeLlama-13b-Instruct/tokenizer.model"},
                   '7I':{'ckpt_dir':"CodeLlama-7b-Instruct/","tokenizer_path":"CodeLlama-7b-Instruct/tokenizer.model"},
                  }
    return model_paths[model]


def extract_code_from_text(text):
    # Regular expression pattern
    pattern = r"```(.*?)```"
    # Find all occurrences of the pattern
    return re.findall(pattern, text, re.DOTALL)[0]


def main(query:str="query.txt",
            unique_id:str='test', out_dir:str='outputs',
            model_type:str="13I", temperature:float=0.2, 
            top_p:float=0.95, max_seq_len:int=512,
            max_batch_size:int=8, max_gen_len: Optional[int]=None):

    out_dir = str(out_dir)
    print(f'loading this query file: {query}')
    # Reading the string from the file
    with open(query, 'r') as file:
        query_text = file.read()

    model_paths = set_model_paths(model=model_type)
    generator = Llama.build(ckpt_dir=model_paths['ckpt_dir'], tokenizer_path=model_paths['tokenizer_path'], 
                            max_seq_len=max_seq_len, max_batch_size=max_batch_size)

    instructions = instructions = [
        [
            {
                "role": "system",
                "content": "Provide code in Python",
            },
            {
                "role": "user",
                "content": query_text
            }
        ]
    ]
    
    print(instructions)
    print('='*69)
    
    results = generator.chat_completion(instructions, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
    print(results)
    # Directory where the file will be saved
    
    print(f'Output dir: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    assert len(results) == 1 # We can add batches later 
    for instruction, result in zip(instructions, results):
        python_code_string = result['generation']['content']
        # Specify the filename
        filename = os.path.join(out_dir, f"{unique_id}_raw.txt")
        # Save the code to a file
        with open(filename, 'w') as file:
            file.write(python_code_string)
        print(f"Raw code saved to: {filename}")
        
        python_code_string = extract_code_from_text(python_code_string)
        filename = os.path.join(out_dir, f"{unique_id}_model.txt")
        with open(filename, 'w') as file:
            file.write(python_code_string)
        print(f"Code saved to: {filename}", flush=True)
    print('JOB DONE')
       
    
if __name__ == "__main__":
    fire.Fire(main)


