import sys
sys.path.append("src")
import re
import os
import glob
import time
import numpy as np
import transformers
from torch import bfloat16
from utils.privit import *
from cfg.constants import *
from utils.print_utils import box_print
from typing import Optional
import requests
import huggingface_hub
from huggingface_hub import InferenceClient
import textwrap
from transformers import AutoTokenizer

def retrieve_base_code(idx):
    """Retrieves base code for quality control."""
    base_network = SEED_NETWORK
    return split_file(base_network)[1:][idx].strip()

def clean_code_from_llm(code_from_llm):
    """Cleans the code received from LLM."""
    code_generator = None
    # Select Correct LLM
    if LLM_MODEL == 'mixtral':
        code_generator = submit_mixtral_local
    elif LLM_MODEL == 'llama3':
        code_generator = submit_llama3_hf
    code_checker_prompt = os.path.join(ROOT_DIR, 'templates/FixedPrompts/validation/code_validation_prompt.txt')
    old_code = ""
    if "```" in code_from_llm:
        try:
            old_code = code_from_llm.split("```")[1]
            if old_code.strip().startswith("python"):
                old_code = '\n'.join(old_code.split("\n")[1:])
        except IndexError:
            print("Failed to extract code block from LLM response.")
            old_code = code_from_llm
    else:
        old_code = code_from_llm
    template_text = ""
    with open(code_checker_prompt, 'r') as file:
        template_text = file.read()
    # Read the info from the prompt
    prompt = template_text.format(old_code.strip())
    box_print("VALIDATING LLM CODE", print_bbox_len=60, new_line_end=False)
    print(prompt)
    verified_code = code_generator(prompt, top_p=0.15, temperature=0.1) 
    print(verified_code)
    return '\n'.join(verified_code.strip().split("```")[1].split('\n')[1:])

def generate_augmented_code(txt2llm, augment_idx, apply_quality_control, top_p, temperature, hugging_face=False):
    """Generates augmented code using Mixtral."""
    box_print("PROMPT TO LLM", print_bbox_len=60, new_line_end=False)
    print(txt2llm)
    if hugging_face is False:
        llm_code_generator = submit_mixtral_local
        qc_func = llm_code_qc
    else:
        if LLM_MODEL == 'mixtral':
            llm_code_generator = submit_mixtral_local
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_hf
        qc_func = llm_code_qc_hf
    if apply_quality_control:
        base_code = retrieve_base_code(augment_idx)
        code_from_llm, generate_text = llm_code_generator(txt2llm, return_gen=True, top_p=top_p, temperature=temperature)
        code_from_llm = qc_func(code_from_llm, base_code, generate_text)
    else:
        code_from_llm = llm_code_generator(txt2llm, top_p=top_p, temperature=temperature)
        # Dealing with No code returned from the LLM
        # Give the LLM 3 Tries to Output correct code
        print("Checking to See if Response was Empty")
        if not code_from_llm or code_from_llm == None:
            print("Response Was Empty")
            for i in range(3):
                code_from_llm = llm_code_generator(txt2llm, top_p=top_p, temperature=temperature) 
                if code_from_llm != "None":
                    break
        print("Response Was Not Empty")
        box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
        print(code_from_llm)
        code_from_llm = clean_code_from_llm(code_from_llm)
    box_print("CODE FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    # This is where I should be fixing the Code Issue if it's None
    return code_from_llm

def extract_note(txt):
    """Extracts note from the part if present."""
    if "# -- NOTE --" in txt:
        note_txt = txt.split('# -- NOTE --')
        return '# -- NOTE --\n' + note_txt[1].strip() + '# -- NOTE --\n'
    return ''

def split_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Regular expression for the pattern
    pattern = r"# --OPTION--"
    parts = re.split(pattern, content)

    return parts

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def llm_code_qc(code_from_llm, base_code, generate_text):
    # TODO: make parameter
    template_path = os.path.join(ROOT_DIR, 'templates/llm_quality_control.txt')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    print("="*120);print(prompt2llm);print("="*120)
    
    res = generate_text(prompt2llm) # clean txt
    code_from_llm = res[0]["generated_text"]
    code_from_llm = '\n'.join(code_from_llm.strip().split("```")[1].split('\n')[1:]).strip()
    return code_from_llm


def llm_code_qc_hf(code_from_llm, base_code, generate_text=None):
    # TODO: make parameter
    fname = np.random.choice(['llm_quality_control_p.txt', 'llm_quality_control_p.txt'])
    template_path = os.path.join(ROOT_DIR, f'templates/{fname}')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    box_print("QC PROMPT TO LLM", print_bbox_len=120, new_line_end=False)
    print(prompt2llm)
    
    code_from_llm = submit_mixtral_local(prompt2llm, max_new_tokens=1500, top_p=0.1, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-v0.1", return_gen=False)
    box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    code_from_llm = clean_code_from_llm(code_from_llm)
    return code_from_llm


def submit_mixtral_hf(txt2mixtral, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2mixtral,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
def submit_mixtral_local(prompt, max_new_tokens=850, temperature=0.2, top_p=0.15, server_url=f"http://{os.getenv('SERVER_HOSTNAME', 'localhost')}:8000/generate", return_gen=False):
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens, # can change to random between 800 - 1000 if needed
        "temperature": temperature,
        "top_p": top_p
    }
    print(os.getenv("SERVER_HOSTNAME", "localhost"))

    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(server_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            output_txt = response.json().get("generated_text", "No output received.")
            print(f'{response.json().get("response_time_sec", "-1")} sec')
            if return_gen is False:
                return output_txt
            else:
                return output_txt, generate_text
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def submit_llama3_hf(txt2llama, 
                     max_new_tokens=1024, 
                     top_p=0.15, 
                     temperature=0.1,                   
                     model_id="google/gemma-2-27b-it",
                     return_gen=False):
    # Randomly set max_new_tokens between 900 and 1300
    max_new_tokens = np.random.randint(900, 1300)
    
    # Set up Hugging Face API key and login
    os.environ['HF_API_KEY'] = "DONT_SCRAPE_ME"  # Replace with your actual key or a method to retrieve it securely
    huggingface_hub.login(new_session=False)
    
    # Create an inference client for the model
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    # Prepare the instructions for the model
    instructions = [
        {
            "role": "user",
            "content": "Provide code in Python\n" + txt2llama,
        },
    ]

    # Load the tokenizer for the model
    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)

    tokenizer_converter.add_special_tokens({'pad_token': '[PAD]'})

    # Manually format the prompt for the model from instructions
    # The original code used apply_chat_template which may not exist in the current tokenizer
    prompt = f"{instructions[0]['role']}: {instructions[0]['content']}\n"

    # Encode the prompt into a tensor suitable for the model
    encoded_prompt = tokenizer_converter.encode(
        prompt, 
        return_tensors='pt', 
        padding=True, 
        truncation=True
    )

    # Generate text from the model
    results = client.text_generation(
        encoded_prompt, 
        max_new_tokens=max_new_tokens, 
        return_full_text=False, 
        temperature=temperature, 
        seed=101
    )

    # Return results based on the specified return type
    if return_gen:
        return results[0], None
    else:
        return results[0]


def submit_llama3_hf(txt2llama, max_new_tokens=1024, top_p=0.15, temperature=0.1,                   # google/gemma-2-27b-it
                     # EleutherAI/gpt-neox-20b
                     # bigcode/starcoder
                     # google/gemma-2-2b-jpn-it
                      model_id="google/gemma-2-2b-jpn-it",
                      return_gen=False):
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [
            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2llama,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False) # Line causing Error
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]


def submit_mixtral(txt2mixtral, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
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
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
    
def mutate_prompts(n=5):
    templates = np.random.choice(glob.glob(f'{ROOT_DIR}/templates/FixedPrompts/*/*.txt'), n)
    for i, template in enumerate(templates):
        path, filename = os.path.split(template)
        with open(template, 'r') as file:
            prompt_text = file.read()
        prompt_text = prompt_text.split("```")[0].strip()
        prompt = "Can you rephrase this text:\n```\n{}\n```".format(prompt_text)
        temp = np.random.uniform(0.01, 0.4)
        if LLM_MODEL == 'mixtral':
            llm_code_generator = submit_mixtral_local
        elif LLM_MODEL == 'llama3':
            llm_code_generator = submit_llama3_hf
        output = llm_code_generator(prompt, temperature=temp).strip()
        if "```" in output:
            output = output.split("```")[0]
        output = output + "\n```python\n{}\n```"
        with open(os.path.join(path, "mutant{}.txt".format(i)), 'w') as file:
            file.write(output)