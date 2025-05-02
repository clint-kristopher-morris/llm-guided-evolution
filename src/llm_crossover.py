import os
import argparse
import random
from cfg.constants import *
from utils.print_utils import box_print
from llm_utils import (split_file, submit_mixtral, submit_mixtral_hf, 
                       llm_code_qc, str2bool, extract_note, generate_augmented_code, 
                       clean_code_from_llm, retrieve_base_code)


def augment_network(input_filename_x, input_filename_y, output_filename,
                    top_p=0.15, temperature=0.1, apply_quality_control=False,
                    hugging_face=False):
    """Augment Python Network Script."""
    # Split the input files
    parts_x = split_file(input_filename_x)
    parts_y = split_file(input_filename_y)
    # Create tuples of parts to be augmented
    parts = [(x, y, idx) for idx, (x, y) in enumerate(zip(parts_x[1:], parts_y[1:]))]
    random.shuffle(parts)
    # Find differing parts
    for x, y, augment_idx in parts:
        if x.strip() != y.strip():
            break

    # Select a template file
    template_fname = random.choice(['crossover.txt', 'crossover_s.txt'])
    template_path = f'{ROOT_DIR}/templates/CrossOver/{template_fname}'
    with open(template_path, 'r') as file:
        template_txt = file.read()

    # Add code to be augmented
    txt2llm = template_txt.format(x.strip(), y.strip())
    # Generate augmented code
    code_from_llm = generate_augmented_code(txt2llm, augment_idx, apply_quality_control,
                                            top_p, temperature, hugging_face=hugging_face)
    # Insert note if present
    temp_txt = parts_x[augment_idx]
    note_txt = extract_note(temp_txt)
    # Update the part with augmented code
    parts_x[augment_idx] = f"\n{note_txt}{code_from_llm}\n"
    # Prepare and write the augmented code to output file
    write_augmented_code(output_filename, parts_x, parts_y)
    box_print(f"Python code saved to {os.path.basename(output_filename)}", print_bbox_len=120, new_line_end=False)
    print('Job done')


def write_augmented_code(output_filename, parts_x, parts_y):
    """Writes the augmented code to the output file."""
    try:
        prompt_log_cross = parts_y[0].split("# --PROMPT LOG--\n")[0]
        prompt_log_cross = f"\n# {'='*10} Start: GeneCrossed\n{prompt_log_cross.strip()}\n# {'='*10} End:\n"
    except IndexError:
        prompt_log_cross = ""

    python_network_txt = prompt_log_cross + '# --OPTION--'.join(parts_x)

    with open(output_filename, 'w') as file:
        file.write(python_network_txt)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Network Script.')

    # Add arguments
    parser.add_argument('input_filename_x', type=str, help='Input file name')
    parser.add_argument('input_filename_y', type=str, help='Input file name')
    parser.add_argument('output_filename', type=str, help='Output file name')
    parser.add_argument('--top_p', type=float, default=0.15, help='Top P value for text generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature value for text generation')
    parser.add_argument('--apply_quality_control', type=str2bool, default=False, help='Use LLM QC')
    parser.add_argument('--hugging_face', type=str2bool, default=False, help='Hugging Face bool')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    augment_network(input_filename_x=args.input_filename_x,
                    input_filename_y=args.input_filename_y,
                    output_filename=args.output_filename,
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    apply_quality_control=args.apply_quality_control,
                    hugging_face=args.hugging_face,
                   )