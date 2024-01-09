import re
import time
import random
import numpy as np
import transformers
from torch import bfloat16
import argparse

from llm_utils import split_file, submit_mixtral


def augment_network(input_filename_x='network.py', 
                    input_filename_y='network_x.py', 
                    output_filename='network_z.py',
                    top_p=0.15, temperature=0.1):
    
    parts_x = split_file(input_filename_x)
    parts_y = split_file(input_filename_y)
    
    parts = [(x, y, idx) for idx, (x, y) in enumerate(zip(parts_x, parts_y))]
    random.shuffle(parts)
    
    for x, y, augment_idx in parts:
        if x.strip() != y.strip():
            break
    
    # TODO: make parameter
    template_path = 'templates/crossover.txt'
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented 
    txt2llm = template_txt.format(x, y)
    
    print("="*120);print(txt2llm);print("="*120)
    
    code_from_llm = submit_mixtral(txt2llm, top_p=top_p, temperature=temperature)
    # clean txt
    code_from_llm = code_from_llm.split('```python')[1].split('```')[0].strip()
    
    if "# -- NOTE --" in parts_x[augment_idx]:
        note_txt = parts_x[augment_idx].split('# -- NOTE --')
        note_txt = '# -- NOTE --\n' + note_txt[1].strip() + '# -- NOTE --\n'
    else:
        note_txt = ''
    
    parts_x[augment_idx] = "\n" + note_txt + code_from_llm + "\n"
    python_network_txt = '# --OPTION--'.join(parts_x)
    print(code_from_llm);print("="*120)
    # Write the text to the file
    
    with open(output_filename, 'w') as file:
        file.write(python_network_txt)
        
    print(f"Python code saved to {output_filename}")
    print('='*120);print('job done');print('='*120)

    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    parser.add_argument('input_filename_x', type=str, help='Input file name')
    parser.add_argument('input_filename_y', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    augment_network(input_filename_x=args.input_filename_x,
                    input_filename_y=args.input_filename_y,
                    output_filename=args.output_filename,
                    top_p=args.top_p, 
                    temperature=args.temperature)