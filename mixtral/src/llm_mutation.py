import re
import time
import numpy as np
import transformers
from torch import bfloat16
import argparse

from llm_utils import split_file, submit_mixtral


def augment_network(input_filename='network.py', output_filename='network_x.py', top_p=0.15, temperature=0.1):
    print(f'Loading {input_filename} code')
    parts = split_file(input_filename)
    augment_idx = np.random.randint(1, len(parts))
    # select code to be augmented randomly 
    code2llm = parts[augment_idx]
    
    if "# -- NOTE --" in code2llm:
        note_txt = code2llm.split('# -- NOTE --')
        note_txt = '# -- NOTE --\n' + note_txt[1].strip() + '# -- NOTE --\n'
    else:
        note_txt = ''
    
    # TODO: make parameter
    fname = np.random.choice(['improvement_xs.txt','improvement_xl.txt','improvement_xp.txt'])
    template_path = f'templates/{fname}'
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented 
    txt2llm = template_txt.format(code2llm)
    print(txt2llm);print("="*120)
    code_from_llm = submit_mixtral(txt2llm, top_p=top_p, temperature=temperature)
    # clean txt
    code_from_llm = code_from_llm.split('```python')[1].split('```')[0].strip()
    parts[augment_idx] = "\n" + note_txt + code_from_llm + "\n"
    python_network_txt = '# --OPTION--'.join(parts)
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
    parser.add_argument('input_filename', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    augment_network(input_filename=args.input_filename,
                    output_filename=args.output_filename,
                    top_p=args.top_p, 
                    temperature=args.temperature)