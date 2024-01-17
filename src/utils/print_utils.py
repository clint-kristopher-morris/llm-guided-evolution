import math
import time

"""
Printing Functions: Move to Other File
"""
def print_population(offspring, global_data):
    box_print(" Poplutation Info:", print_bbox_len=60, new_line_end=False)
    print(f'🧬 Poplutation Size 🧬: {len(offspring)}')
    for ind in offspring:
        gene_id = ind[0]
        print(f'Gene: {gene_id}')
        if gene_id in global_data:
            print_job_info(global_data[gene_id], short=True)

def print_scores(population, fitness_weights):
    num_objectives = len(fitness_weights)
    objective_scores = [[] for _ in range(num_objectives)]

    # Collect scores for each objective
    for ind in population:
        for i, f in enumerate(ind.fitness.values):
            objective_scores[i].append(f)

    # Calculate and print stats for each objective
    box_print("SCORES", print_bbox_len=110, new_line_end=True)
    for i in range(num_objectives):
        fits = objective_scores[i]
        fits = [x for x in objective_scores[i] if math.isfinite(x)]
        length = len(fits)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        direction = "Maximize" if fitness_weights[i] > 0 else "Minimize"
        
        print(f"Objective {i+1} ({direction}):")
        print(f"  Min: {min(fits)}")
        print(f"  Max: {max(fits)}")
        print(f"  Avg: {mean}")
        print(f"  Std: {std}")
        print()

def box_print(txt, print_bbox_len=110, new_line_end=True):
    # just for logging 
    def replace_middle(v, x):
        start_pos = (len(v) - len(x)) // 2
        return v[:start_pos] + x + v[start_pos + len(x):]
    
    v = "*" + " " * (print_bbox_len - 2) + "*"
    end = '\n' if new_line_end else ''
    print_result = "\n" + "*" * print_bbox_len + "\n" + replace_middle(v, txt) + "\n" + "*" * print_bbox_len + end
    print(print_result, flush=True)
    
def print_job_info(job_dict, short=False):
    print(f"\t‣ Fitness: {job_dict['fitness']}, Submission Flag: {job_dict['sub_flag']}")
    print(f"\t‣ Runtime: {round((time.time()-job_dict['start_time'])/60)} min, Status: {job_dict['status']}")
    if short is False:
        try:
            print(f"\t‣ LLM Job-ID: {job_dict['job_id']}, Model Job-ID: {job_dict['results_job']}")
        except:
            print(f"\t‣ {job_dict}")
    else:
        print(f"\t‣ LLM Job-ID: {job_dict['job_id']}")