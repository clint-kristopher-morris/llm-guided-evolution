import re
import time
import numpy as np
import transformers
from torch import bfloat16
import argparse
from utils.print_utils import box_print
from llm_utils import split_file, submit_mixtral, sibmit_mixtral_hf, llm_code_qc, str2bool, generate_augmented_code, extract_note, clean_code_from_llm, retrieve_base_code


def augment_network(input_filename='network.py', output_filename='network_x.py',
                    top_p=0.15, temperature=0.1, apply_quality_control=False, hugging_face=False):
    print(f'Loading {input_filename} code')
    parts = split_file(input_filename)
    augment_idx = np.random.randint(1, len(parts))
    # select code to be augmented randomly 
    code2llm = parts[augment_idx]
    

    # TODO: make parameter
    fname = np.random.choice(['improvement_xs.txt','improvement_xlp.txt','improvement_xlpc.txt',
                              'improvement_xl.txt','improvement_xlpl.txt','improvement_xp2.txt',
                              'improvement_xp.txt','improvement_xm.txt'])
    template_path = f'/gv1/projects/AI_Surrogate/dev/clint/CodeLLama/codellama/mixtral/templates/{fname}'
    with open(template_path, 'r') as file:
        template_txt = file.read()
            
    # add code to be augmented 
    txt2llm = template_txt.format(code2llm)
    code_from_llm = generate_augmented_code(txt2llm, augment_idx-1, apply_quality_control,
                                            top_p, temperature, hugging_face=hugging_face)
    
    note_txt = extract_note(code2llm)
    parts[augment_idx] = f"\n{note_txt}{code_from_llm}\n"
    prompt_log = f'# Parent Prompt: {fname} Root Code: {input_filename}\n'
    python_network_txt = prompt_log + '# --OPTION--'.join(parts)

    # Write the text to the file
    with open(output_filename, 'w') as file:
        file.write(python_network_txt)
        
    box_print(f"Python code saved to {output_filename}", print_bbox_len=120, new_line_end=False)
    print('Job Done')

    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    parser.add_argument('input_filename', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')
    parser.add_argument('--hugging_face', type=str2bool, default=False, help='Hugging Face bool')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    augment_network(input_filename=args.input_filename,
                    output_filename=args.output_filename,
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                    hugging_face=args.hugging_face,
                   )