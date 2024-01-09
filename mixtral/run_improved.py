import os
import time
import string
import random
import pickle
import subprocess
import numpy as np
from deap import base, creator, tools
from deap.tools import HallOfFame


FITNESS_WEIGHTS = (1.0, -1.0)
INVALID_FITNESS_MAX = tuple([float(x*np.inf*-1) for x in FITNESS_WEIGHTS])
TMP_FITNESS = tuple([int(x*9999999999*-1) for x in FITNESS_WEIGHTS])
LLM_GPU = 'NVIDIAA100-SXM4-80GB'

PYTHON_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:1
#SBATCH -C "QuadroP4000|QuadroRTX4000|GeForceGTX1080Ti|GeForceGTX1080|TeslaV100-PCIE-32GB|TeslaV100S-PCIE-32GB"
#SBATCH --mem 8G
#SBATCH -c 4
echo "Launching AIsurBL"
hostname

# Load GCC version 9.2.0
module load gcc/13.2.0
# module load cuda/11.8

# Activate Conda environment
source /opt/apps/Module/anaconda3/2021.11/bin/activate mix
conda info

# Set the TOKENIZERS_PARALLELISM environment variable if needed
export TOKENIZERS_PARALLELISM=false

# Run Python script
{}
"""

LLM_BASH_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:2
#SBATCH -C "{}"
#SBATCH --mem 32G
#SBATCH -c 1
echo "Launching AIsurBL"
hostname

# Load GCC version 9.2.0
module load gcc/13.2.0
# module load cuda/11.8

# Activate Conda environment
source /opt/apps/Module/anaconda3/2021.11/bin/activate mix
conda info

# Set the TOKENIZERS_PARALLELISM environment variable if needed
export TOKENIZERS_PARALLELISM=false

# Run Python script
{}
"""


def box_print(txt, print_bbox_len=110, new_line_end=True):
    # just for logging 
    def replace_middle(v, x):
        start_pos = (len(v) - len(x)) // 2
        return v[:start_pos] + x + v[start_pos + len(x):]
    
    v = "*" + " " * (print_bbox_len - 2) + "*"
    end = '\n' if new_line_end else ''
    print_result = "\n" + "*" * print_bbox_len + "\n" + replace_middle(v, txt) + "\n" + "*" * print_bbox_len + end
    print(print_result, flush=True)

# was write_bash_script
def write_bash_script(input_filename_x='./../sota/ExquisiteNetV2/network.py',
                      input_filename_y=None,
                      output_filename='./../sota/ExquisiteNetV2/models/network_x.py',
                      gpu='TeslaV100-PCIE-32GB',
                      python_file='src/llm_mutation.py', 
                      top_p=0.1, temperature=0.2):
    
    # gpu='NVIDIAA100-SXM4-80GB'
    if python_file=='src/llm_mutation.py':
        python_runline = f'python {python_file} {input_filename_x} {output_filename} --top_p {top_p} --temperature {temperature}'
    elif python_file=='src/llm_crossover.py':
        temp_text = f"{python_file} {input_filename_x} {input_filename_y} {output_filename}"
        python_runline = f'python {temp_text} --top_p {top_p} --temperature {temperature}' 
    else:
        raise ValueError("Invalid python_file argument")

    bash_script_content = LLM_BASH_SCRIPT_TEMPLATE.format(gpu, python_runline)
    return bash_script_content


def create_bash_file(file_path, **kwargs):
    bash_script_content = write_bash_script(**kwargs)
    # Extract the directory from the file path
    directory = os.path.dirname(file_path)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Write the file
    with open(file_path, 'w') as file:
        file.write(bash_script_content)
    print(f"Bash script saved to {file_path}", flush=True)


def submit_bash(file_path, **kwargs):
    """ This should be general for subbing anything and returning:
        successful_sub_flag 
        job_id
    """
    create_bash_file(file_path, **kwargs)
    result = subprocess.run(["bash", file_path], capture_output=True, text=True)

    if result.returncode == 0:
        print("Script submitted successfully.\nOutput:", result.stdout, flush=True)
        successful_sub_flag = True
        job_id = result.stdout.split('job ')[-1].strip()
    else:
        print("Failed to submit script.\nError:", result.stderr, flush=True)
        successful_sub_flag = False
        job_id = None

    return successful_sub_flag, job_id


def check4job_completion(job_id, check_interval=60, timeout=3600):
    """
    Check for the completion of a job by searching for its output file and scanning for errors.

    Parameters:
    job_id (str): The job ID to check.
    check_interval (int): Time in seconds between checks.
    timeout (int): Maximum time in seconds to wait for job completion.

    Returns:
    bool: True if job completed successfully, False otherwise.
    """
    start_time = time.time()
    output_file = f'slurm-{job_id}.out'

    while True:
        # Check if the timeout is reached
        if time.time() - start_time > timeout:
            print("Timeout reached while waiting for job completion.")
            return False

        # Check if the output file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as file:
                contents = file.read()
                # Check for error indicators in the file
                if "traceback" in contents.lower():
                    print("Error found in LLM job output.", flush=True)
                    return False
                elif "job done" in contents.lower():
                    print("LLM Job completed successfully.", flush=True)
                    return True
                else:
                    pass

        # Wait for some time before checking again
        time.sleep(check_interval)
        print(f'waiting on check4job_completion LLM job:{job_id} time:{round(time.time() - start_time)}', flush=True)
        
        
def generate_random_string(length=20):
    # Define the characters that can be used in the string
    characters = string.ascii_letters + string.digits
    # Generate a random string of specified length
    random_string = ''.join(random.choice(characters) for i in range(length))
    random_string = 'xXx'+random_string
    return random_string


def create_individual(container, temp_min=0.05, temp_max=0.4):
    box_print("Create Individual", print_bbox_len=60, new_line_end=False)
    out_dir = str(GENERATION)
    gene_id = generate_random_string(length=24)
    # Select prompte and temp
    temperature = round(random.uniform(temp_min, temp_max), 2)
    # Assign a file path and name for the model creation bash
    file_path = os.path.join(out_dir, f'{gene_id}.sh')
    successful_sub_flag, job_id = submit_bash(file_path, 
                                              input_filename_x='./../sota/ExquisiteNetV2/network.py',
                                              output_filename=f'./../sota/ExquisiteNetV2/models/network_{gene_id}.py',
                                              gpu=LLM_GPU,
                                              python_file='src/llm_mutation.py', 
                                              top_p=0.1, temperature=temperature)
    # Log data
    GLOBAL_DATA[gene_id] = {'sub_flag':successful_sub_flag, 'job_id':job_id, 
                            'status':'subbed file', 'fitness':None, 'start_time':time.time()}
    individual = container([gene_id])  # Assign a file ID
    
    if successful_sub_flag:
        print(f'Checking for job completion: {job_id} for {gene_id}', flush=True)
        job_done = check4job_completion(job_id)
        print(f'Model files for {gene_id} are loaded') if job_done else print(f'Error loading model files for {gene_id}', flush=True)
    # return individual,
    return individual


def submit_run(gene_id):
    def write_bash_script_py(gene_id, train_file='./../sota/ExquisiteNetV2/train.py'):
        tmp = "-data cifar10 -end_lr 0.001 -seed 21 -val_r 0.2 -amp"
        # python_runline = f'python {train_file} -bs 384 -network "models.network_{gene_id}" {tmp}'
        python_runline = f'python {train_file} -bs 400 -epoch 2 -network "models.network_{gene_id}" {tmp}'
        bash_script_content = PYTHON_BASH_SCRIPT_TEMPLATE.format(python_runline)
        return bash_script_content

    # This is for subbing the python code
    def create_bash_file_py(file_path, gene_id, **kwargs):
        bash_script_content = write_bash_script_py(gene_id, **kwargs)
        with open(file_path, 'w') as file:
            file.write(bash_script_content)
        print(f"Bash script saved to {file_path}")

    def submit_bash_py(file_path, gene_id, **kwargs):
        create_bash_file_py(file_path, gene_id, **kwargs)
        result = subprocess.run(["sbatch", file_path], capture_output=True, text=True)
        if result.returncode == 0:
            print("Script submitted successfully.\nOutput:", result.stdout)
            successful_sub_flag = True
            job_id = result.stdout.split('job ')[-1].strip()
        else:
            print("Failed to submit script.\nError:", result.stderr)
            successful_sub_flag = False
            job_id = None
        return successful_sub_flag, job_id
    
    out_dir = str(GENERATION)
    file_path = os.path.join(out_dir, f'{gene_id}_model.sh')
    successful_sub_flag, job_id = submit_bash_py(file_path, gene_id)
    GLOBAL_DATA[gene_id]['status'] = 'running eval'
    GLOBAL_DATA[gene_id]['results_job'] = job_id
    print(f'Running py file for {gene_id}, {job_id}')

    
def evalModel(individual):
    gene_id = individual[0]
    # Initially, we don't have a fitness value
    return None


def check4model2run(gene_id):
    # model_path = os.path.join(str(GENERATION), f'{gene_id}_model.txt')
    print(f'Checking for: ./../sota/ExquisiteNetV2/models/network_{gene_id}.py')
    model_path = f'./../sota/ExquisiteNetV2/models/network_{gene_id}.py'
    if os.path.exists(model_path):
        if GLOBAL_DATA[gene_id]['status'] != 'running eval':
            submit_run(gene_id)
            
            
def check4results(gene_id):
    # TODO: fix this to check for my results also save the results from the train.py into a file
    def check4error(gene_id):
        job_id = GLOBAL_DATA[gene_id]['results_job']
        output_file = f'slurm-{job_id}.out'
        # Check if the output file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as file:
                contents = file.read()
                # Check for error indicators in the file
                if "traceback" in contents.lower():
                    print("Error found in job output.")
                    return False
                elif "job done" in contents.lower():
                    print("Job completed successfully.", flush=True)
                    return True
                else:
                    pass
        return None
                
    job_done = check4error(gene_id)
    if job_done is True:
        out_dir = str(GENERATION)
        # The job saves the model results to a file f'{gene_id}_results.txt'
        # results_path = os.path.join(out_dir, f'{gene_id}_results.txt')
        results_path = f'./../sota/ExquisiteNetV2/results/{gene_id}_results.txt'
        with open(results_path, 'r') as file:
            results = file.read()
        results = results.split(',')
        fitness = [float(r.strip()) for r in results]
        
        # TODO: get all features later
        fitness = [fitness[0], fitness[1]]
        fitness = tuple(fitness)
        
        GLOBAL_DATA[gene_id]['status'] = 'completed'
        GLOBAL_DATA[gene_id]['fitness'] = fitness
        print(f'Model from gene: {gene_id} evaluated')
    elif job_done is False:
        GLOBAL_DATA[gene_id]['status'] = 'completed'
        GLOBAL_DATA[gene_id]['fitness'] = INVALID_FITNESS_MAX
        print(f'Model from gene: {gene_id} failed to run')
    else:
        print('Job has not finished running yet...', flush=True)
        pass
        

# TODO: if llm fails it forces a max wait time
def check_and_update_fitness(population, timeout=3600*20, loop_delay=60*2.5):
    # Set a timeout for each job
    box_print("SUBMITTING MODELS CREATED BY LLM")
    count = 0
    while True:
        box_print(f"Checking Model Runs: {count}", print_bbox_len=60, new_line_end=False)
        all_done = True
        for ind in population:
            gene_id = ind[0]
            # check if failed job
            if GLOBAL_DATA[gene_id]['sub_flag']==False:
                ind.fitness.values = INVALID_FITNESS_MAX # Max error
                GLOBAL_DATA[gene_id]['status'] == "completed"
            if ind.fitness.values == TMP_FITNESS:  # If fitness not assigned
                # check for gene_id_model.txt file
                if GLOBAL_DATA[gene_id]['status'] == 'subbed file':
                    # here we generate/sub model.txt file
                    # also we assign GLOBAL_DATA[gene_id]['status'] = 'running eval' in check4model2run
                    check4model2run(gene_id) 
                if GLOBAL_DATA[gene_id]['status'] == 'running eval':
                    print(f'Checking for results for: {gene_id}')
                    non_error_flag = check4results(gene_id) # here we look for the results file for gene_id
                    if non_error_flag == False:
                        # TODO: change FAILED: TIMEOUT to "LLM FAILURE"
                        print(f"LLM failed for gene ID {gene_id}")
                        ind.fitness.values = INVALID_FITNESS_MAX 
                        GLOBAL_DATA[gene_id]['status'] = 'completed'
                
                
                if GLOBAL_DATA[gene_id]['status'] == "completed":
                    # Process results and assign fitness
                    fitness_tuple = GLOBAL_DATA[gene_id]['fitness']  # Implement this function
                    ind.fitness.values = fitness_tuple
                elif time.time() - GLOBAL_DATA[gene_id]['start_time'] > timeout:
                    print(f"Timeout for gene ID {gene_id}")
                    ind.fitness.values = INVALID_FITNESS_MAX 
                    GLOBAL_DATA[gene_id]['status'] = 'FAILED: TIMEOUT'
                else:
                    # print(f"Still Waiting On: \n\t(Gene:{gene_id}, LLM_Job:{GLOBAL_DATA['job_id']}, Model_Job:{GLOBAL_DATA['results_job']} Status:{''}")
                    print(f"Still Waiting On: Gene:{gene_id}")
                    print(GLOBAL_DATA[gene_id])
                    all_done = False  # Some jobs are still running
        if all_done:
            box_print("Evalutated All Genes", print_bbox_len=80)
            break  # All jobs are done or timed out
        time.sleep(loop_delay)  # Wait some time before checking again
        count+=1
        
        
# Custom crossover function
def customCrossover(ind1, ind2):
    def combine_elements(ind1, ind2, temp_min=0.05, temp_max=0.4):
        """
        Combine elements of two individuals to create a new individual.
        Parameters:
        ind1, ind2 (list): The parent individuals.
        Returns:
        str: The gene ID of the new individual.
        """
        global GLOBAL_DATA
        out_dir = str(GENERATION)
        # Retrieve gene IDs from the individuals
        gene_id_1 = ind1[0]
        gene_id_2 = ind2[0]
        # Generate the crossover query
        print(f'Mating: {gene_id_1} and {gene_id_2}')
        temperature = round(random.uniform(temp_min, temp_max), 2)
        # Generate a new gene ID for the offspring
        new_gene_id = generate_random_string(length=24)
        # Create the bash file for the new job
        file_path = os.path.join(out_dir, f'{new_gene_id}.sh')
        successful_sub_flag, job_id = successful_sub_flag, job_id = submit_bash(file_path, 
                                          input_filename_x=f'./../sota/ExquisiteNetV2/models/network_{gene_id_1}.py',
                                          input_filename_y=f'./../sota/ExquisiteNetV2/models/network_{gene_id_2}.py',
                                          output_filename=f'./../sota/ExquisiteNetV2/models/network_{new_gene_id}.py',
                                          gpu=LLM_GPU,
                                          python_file='src/llm_crossover.py', 
                                          top_p=0.1, temperature=temperature)

        # Update global data for the new individual
        GLOBAL_DATA[new_gene_id] = {'sub_flag':successful_sub_flag, 'job_id':job_id, 
                                    'status':'subbed file', 'fitness':None, 'start_time':time.time()}
        
        if successful_sub_flag:
            print(f'Checking for job completion - cross: {job_id} for {new_gene_id}')
            job_done = check4job_completion(job_id)
            print(f'Model files for {new_gene_id} are loaded') if job_done else print(f'Error loading model files for {new_gene_id}!!!\n\n')

        failed_process = True if (successful_sub_flag is False) or (job_done is False) else False
        # Return the new gene ID
        return new_gene_id, failed_process
    
    global GLOBAL_DATA
    
    new_gene_id1, failed_process1 = combine_elements(ind1, ind2)
    new_gene_id2, failed_process2 = combine_elements(ind2, ind1)
    
    if failed_process1 is False:
        if ind1[0] in GLOBAL_DATA.keys():
            del GLOBAL_DATA[ind1[0]]
        ind1[0] = new_gene_id1
        offspring1 = creator.Individual([new_gene_id1])
        print(f'Mated: {new_gene_id1}')
    else:
        print(f'Failed Mate: {new_gene_id1}')
        offspring1 = ind1
        del GLOBAL_DATA[new_gene_id1]
        
    if failed_process2 is False:
        if ind2[0] in GLOBAL_DATA.keys():
            del GLOBAL_DATA[ind2[0]]
        ind2[0] = new_gene_id2
        offspring2 = creator.Individual([new_gene_id2])
        print(f'Mated: {new_gene_id2}')
    else:
        print(f'Failed Mate: {new_gene_id2}')
        offspring2 = ind2
        del GLOBAL_DATA[new_gene_id2]
        
    return offspring1, offspring2


def customMutation(individual, indpb, temp_min=0.05, temp_max=0.4):
    """ Custom mutation function that randomly changes the temperature parameter of the individual's task and assigns a new ID.
    Parameters:
    individual (list): The individual to be mutated.
    indpb (float): The probability of mutating each gene.
    Returns:
    tuple: The mutated individual.
    """
    # Check if mutation occurs (based on the mutation probability)
    # if random.random() < indpb: # TODO: connect this to temp
    out_dir = str(GENERATION)
    old_gene_id = individual[0]
    # Generate a new gene ID
    new_gene_id = generate_random_string(length=24)
    print(f'Mutating: {old_gene_id} and replaceing with: {new_gene_id}')
    # Name of the sh bash file
    file_path = os.path.join(str(GENERATION), f'{new_gene_id}.sh')
    temperature = round(random.uniform(temp_min, temp_max), 2)
    successful_sub_flag, job_id = submit_bash(file_path, 
                                              input_filename_x= f'./../sota/ExquisiteNetV2/models/network_{old_gene_id}.py',
                                              output_filename=f'./../sota/ExquisiteNetV2/models/network_{new_gene_id}.py',
                                              gpu=LLM_GPU,
                                              python_file='src/llm_mutation.py', 
                                              top_p=0.1, temperature=temperature)
    
    # Update the individual with the new gene ID
    # individual[0] = new_gene_id
    # Update the global data with the new task
    GLOBAL_DATA[new_gene_id] = {'sub_flag':successful_sub_flag, 'job_id':job_id, 
                                'status':'subbed file', 'fitness':None, 'start_time':time.time()}

    if successful_sub_flag:
        print(f'Checking for job completion Mutate: {job_id} for {new_gene_id}')
        job_done = check4job_completion(job_id)
        print(f'Model files for {new_gene_id} are loaded') if job_done else print(f'Error loading model files for {new_gene_id}')

    failed_process = True if (successful_sub_flag is False) or (job_done is False) else False
    
    if failed_process is False:
        individual[0] = new_gene_id
        individual = creator.Individual([new_gene_id])
        if old_gene_id in GLOBAL_DATA.keys():
            del GLOBAL_DATA[old_gene_id]
        print(f'Mutated: {new_gene_id}')
    else:
        print(f'Failed Mutate: {new_gene_id}')
        del GLOBAL_DATA[new_gene_id]
    return individual


def print_scores(fits, population):
    length = len(population)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    print(f"  Min: {min(fits)}")
    print(f"  Max: {max(fits)}")
    print(f"  Avg: {mean}")
    print(f"  Std: {std}")
    
    
def remove_duplicates(population):
    unique_individuals = []
    seen_chromosomes = set()

    for individual in population:
        # Convert chromosome to a tuple since lists are not hashable
        chromosome = tuple(individual)  
        if chromosome not in seen_chromosomes:
            unique_individuals.append(individual)
            seen_chromosomes.add(chromosome)

    return unique_individuals


# --- Checkpoint Functions --- #
def save_checkpoint(gen, folder_name="checkpoints"):
    os.makedirs(folder_name, exist_ok=True)
    checkpoint_data = {
        "GLOBAL_DATA": GLOBAL_DATA,
        "GLOBAL_DATA_HIST": GLOBAL_DATA_HIST,
        "population": population,
        "hof": hof,
    }
    filename = os.path.join(folder_name, f'checkpoint_gen_{gen}.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(checkpoint_data, file)
    print(f"Checkpoint saved as {filename}")

    
def load_checkpoint(folder_name="checkpoints", checkpoint_file=None):
    if not os.path.exists(folder_name):
        return None, None
    if checkpoint_file is None:
        checkpoint_files = sorted(os.listdir(folder_name), reverse=True)
        checkpoint_file = checkpoint_files[0] if checkpoint_files else None
    if checkpoint_file:
        filepath = os.path.join(folder_name, checkpoint_file)
        with open(filepath, 'rb') as file:
            checkpoint_data = pickle.load(file)
        print(f"Loaded checkpoint from {filepath}")
        return checkpoint_data, int(checkpoint_file.split('_')[2].split('.')[0])
    return None, None


def true_nsga2(pop, k):
    pop = tools.selNSGA2(pop, len(pop)) # 10 diff
    new_pop = tools.selTournamentDCD(pop, k) # mults of 4
    return new_pop

# Define the problem
creator.create("FitnessMulti", base.Fitness, weights=FITNESS_WEIGHTS)  # Adjust weights as needed
creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)

# Initialize the toolbox
toolbox = base.Toolbox()
toolbox.register("individual", create_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalModel)
toolbox.register("mate", customCrossover)
toolbox.register("mutate", customMutation, indpb=0.2)
toolbox.register("select", true_nsga2)

# Change epoch, change loop check time to 60*30,

# --- Parameters --- #
# GENERATION = 0
# GLOBAL_DATA = {}
# GLOBAL_DATA_HIST = {}
# num_generations = 30  # Number of generations
# start_population_size = 16  # Size of the population
# population_size = 12
# crossover_probability = 0.25  # Probability of mating two individuals
# mutation_probability = 0.4   # Probability of mutating an individual
# num_elites = 16
# hof_size = 100

GENERATION = 0
GLOBAL_DATA = {}
GLOBAL_DATA_HIST = {}
num_generations = 30  # Number of generations
start_population_size = 10  # Size of the population
population_size = 4
crossover_probability = 0.99  # Probability of mating two individuals
mutation_probability = 0.99   # Probability of mutating an individual
num_elites = 4
hof_size = 100

# Main Evolution Loop
if __name__ == "__main__":
    checkpoint, start_gen = load_checkpoint()
    if checkpoint:
        box_print("LOADING CHECKPOINT")
        GLOBAL_DATA = checkpoint["GLOBAL_DATA"]
        GLOBAL_DATA_HIST = checkpoint["GLOBAL_DATA_HIST"]
        population = checkpoint["population"]
        hof = checkpoint["hof"]
    else:
        # Create an initial population
        box_print("CREATING POPULATION FROM SEED CODE")
        population = toolbox.population(n=start_population_size)
        start_gen = 0
        hof = tools.HallOfFame(hof_size)

    # Evaluate the entire population
    for ind in population:
        ind.fitness.values = TMP_FITNESS
    check_and_update_fitness(population)
    # Evolution
    for gen in range(start_gen, num_generations):
        box_print(f"STARTING GENERATION: {gen}")
        # Select the next generation's parents
        elites = tools.selSPEA2(population, num_elites)
        # elites = tools.selBest(population, num_elites)
        # Select the next generation's parents
        offspring = toolbox.select(population, population_size)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        GLOBAL_DATA_HIST.update(GLOBAL_DATA.copy())
        
        # Remove individuals with placeholder fitness
        offspring = [ind for ind in offspring if ind.fitness.values != INVALID_FITNESS_MAX]

        # Apply crossover and mutation on the offspring
        box_print("Mating", print_bbox_len=60, new_line_end=False)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_probability:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        box_print("Mutating", print_bbox_len=60, new_line_end=False)
        # Might need to re-init here
        for mutant in offspring:
            if random.random() < mutation_probability:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Add elites back to offspring. Usually before the mute and cross but in this case we save them
        offspring.extend(elites)
        # After merging the offspring and the elites
        offspring = remove_duplicates(offspring)
        elites_keys = [k[0] for k in elites]
        for k in elites_keys:
            # Bring back the elite history
            GLOBAL_DATA[k] = GLOBAL_DATA_HIST[k]

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind in offspring:
            # assign placeholder to all so I can check them all at once
            ind.fitness.values = TMP_FITNESS 

        GLOBAL_DATA_HIST.update(GLOBAL_DATA.copy())
        check_and_update_fitness(offspring)
        GLOBAL_DATA_HIST.update(GLOBAL_DATA.copy())
        # Replace the old population with the offspring
        population[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits1 = [ind.fitness.values[0] for ind in population]
        fits2 = [ind.fitness.values[1] for ind in population]

        print_scores(fits1, population)
        print_scores(fits2, population)
        hof.update(population)

        save_checkpoint(gen)

    print("-- End of Evolution --")
    best_ind = tools.selBest(population, 1)[0]
    print(f"Best Individual: {best_ind}")
    print(f"Best Fitness: {best_ind.fitness.values}")
