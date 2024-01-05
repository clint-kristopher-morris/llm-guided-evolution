import subprocess
import os
import string
import time
import random
from deap import base, creator, tools
from deap.tools import HallOfFame
import pickle


def write_bash_script(gpu='TeslaV100S-PCIE-32GB', fname='queries/py/query_instruct.py', max_seq_len=1024, temp=0.2, unique_id='test', out_dir='./', query=None):
    bash_script_content = f"""#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:2
#SBATCH -C "{gpu}"
#SBATCH --mem 32G
#SBATCH -c 1
echo "launching AIsurBL"
hostname
# module load anaconda3/2020.07 2021.11
module load cuda/11.0

source /opt/apps/Module/anaconda3/2021.11/bin/activate
conda activate REMAPS
conda info

torchrun --nproc_per_node 2 {fname} --max_seq_len {max_seq_len} --query {query} --temperature {temp} --unique_id {unique_id} --out_dir {out_dir}
"""
    return bash_script_content

def create_bash_file(file_path, query, **kwargs):
    bash_script_content = write_bash_script(query=query, **kwargs)
    with open(file_path, 'w') as file:
        file.write(bash_script_content)
    print(f"Bash script saved to {file_path}")

def submit_bash(query, file_path, **kwargs):
    create_bash_file(file_path, query, **kwargs)
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


def check4job_completion(job_id, check_interval=20, timeout=3600):
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
                    print("Error found in job output.")
                    return False
                elif "job done" in contents.lower():
                    print("Job completed successfully.")
                    return True
                else:
                    pass

        # Wait for some time before checking again
        time.sleep(check_interval)
        
        
def generate_random_string(length=20):
    # Define the characters that can be used in the string
    characters = string.ascii_letters + string.digits
    # Generate a random string of specified length
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string


def create_individual(container, creation_queries_path="queries/general/create_prompts.txt"):
    print('='*60)
    out_dir = str(GENERATION)
    gene_id = generate_random_string(length=24)
    # Load prompt tempate 
    with open(creation_queries_path, 'r') as file:
        creation_queries_text = file.read()
    # Select prompte and temp
    creation_query = random.choice(creation_queries_text.split('\n'))
    temp = round(random.uniform(0.1, 0.4), 2)
    # Create a query/prompt text file that will be loaded by llama code
    query_path = os.path.join(out_dir, f'{gene_id}.txt')
    with open(query_path, 'w') as file:
        file.write(creation_query)
    # Assign a file path and name for the model creation bash
    file_path = os.path.join(out_dir, f'{gene_id}.sh')
    
    successful_sub_flag, job_id = submit_bash(query_path, file_path, gpu='TeslaV100S-PCIE-32GB', 
                                              fname='queries/py/query_instruct.py', max_seq_len=2048, temp=temp,
                                              unique_id=gene_id, out_dir=out_dir)
    # Log data
    GLOBAL_DATA[gene_id] = {'sub_flag':successful_sub_flag, 'job_id':job_id, 
                            'status':'subbed file', 'fitness':None, 'start_time':time.time()}
    individual = container([gene_id])  # Assign a file ID
    
    if successful_sub_flag:
        print(f'Checking for job completion: {job_id} for {gene_id}')
        job_done = check4job_completion(job_id)
        if job_done:
            print(f'Model files for {gene_id} are loaded')
        else:
            print(f'Error loading model files for {gene_id}')
   
    # return individual,
    return individual

def write_bash_script_py(py_file_path):
    bash_script_content = f"""#!/bin/bash
#SBATCH --job-name=AIsur_x1
#SBATCH -t 8-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem 8G
#SBATCH -c 1
echo "launching AIsurBL"
hostname
# module load anaconda3/2020.07 2021.11
module load cuda/11.0

source /opt/apps/Module/anaconda3/2021.11/bin/activate
conda activate REMAPS
conda info

python {py_file_path}
"""
    return bash_script_content

def create_bash_file_py(file_path, py_file_path, **kwargs):
    bash_script_content = write_bash_script_py(py_file_path, **kwargs)
    with open(file_path, 'w') as file:
        file.write(bash_script_content)
    print(f"Bash script saved to {file_path}")

def submit_bash_py(py_file_path, file_path, **kwargs):
    create_bash_file_py(file_path, py_file_path, **kwargs)
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

def submit_run(gene_id):
    def create_py_file(gene_id, run_file_template_path="queries/general/run_file_template.txt", 
                       base_model_path="queries/general/base_model.txt"):
        out_dir = str(GENERATION)
        # load model txt
        model_path = os.path.join(out_dir, f'{gene_id}_model.txt')
        with open(model_path, 'r') as file:
            model_txt = file.read()
        # init results path:
        results_path = os.path.join(out_dir, f'{gene_id}_results.txt')
        # load template file
        with open(run_file_template_path, 'r') as file:
            run_file_template = file.read()
        temp_list = run_file_template.split('# Step 2: Classifier Model')
        
        # Save Jaccard Score
        with open(base_model_path, 'r') as file:
            code_base = file.read()
        code_model_x = model_txt
        # Convert each snippet to a set of lines (strip whitespace and remove empty lines)
        set_1 = set(line.strip() for line in code_base.split('\n') if line.strip())
        set_2 = set(line.strip() for line in code_model_x.split('\n') if line.strip())
        intersection = set_1.intersection(set_2) # Calculate intersection and union
        union = set_1.union(set_2)
        jaccard_similarity = len(intersection) / len(union) # Compute Jaccard Similarity
        
        file_path = os.path.join(out_dir, f'{gene_id}_jaccard_score.txt')
        with open(file_path, 'w') as file:
            file.write(str(round(jaccard_similarity, 6)))
        
        run_file_text = temp_list[0] + f"results_path='{results_path}'\n" + model_txt + temp_list[1]
        # Write the text to the file
        py_file_path = os.path.join(out_dir, f'{gene_id}_model.py')
        with open(py_file_path, 'w') as file:
            file.write(run_file_text)
        print(f"Python file '{py_file_path}' created successfully.", flush=True)
        time.sleep(2)
        return py_file_path
    py_file_path = create_py_file(gene_id)
    sh_file_path = py_file_path.replace('.py','.sh')
    successful_sub_flag, job_id = submit_bash_py(py_file_path, sh_file_path)
    GLOBAL_DATA[gene_id]['status'] = 'running eval'
    GLOBAL_DATA[gene_id]['results_job'] = job_id
    print(f'Running py file for {gene_id}, {job_id}')

    
def evalModel(individual):
    gene_id = individual[0]
    # Initially, we don't have a fitness value
    return None


def check4model2run(gene_id):
    model_path = os.path.join(str(GENERATION), f'{gene_id}_model.txt')
    if os.path.exists(model_path):
        if GLOBAL_DATA[gene_id]['status'] != 'running eval':
            submit_run(gene_id)
            
            
def check4results(gene_id):
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
        results_path = os.path.join(out_dir, f'{gene_id}_results.txt')
        with open(results_path, 'r') as file:
            results = file.read()
        results = results.split(',')
        fitness = [float(r.strip()) for r in results]
        # Replacing F1 with Jaccard Score
        file_path = os.path.join(out_dir, f'{gene_id}_jaccard_score.txt')
        with open(file_path, 'r') as file:
            result_jaccard = file.read()
        js = float(result_jaccard.strip())
        fitness[-1] = js
        fitness = tuple(fitness)
        
        GLOBAL_DATA[gene_id]['status'] = 'completed'
        GLOBAL_DATA[gene_id]['fitness'] = fitness
        print(f'Model from gene: {gene_id} evaluated')
    elif job_done is False:
        GLOBAL_DATA[gene_id]['status'] = 'completed'
        GLOBAL_DATA[gene_id]['fitness'] = (-float('inf'), float('inf'))
        print(f'Model from gene: {gene_id} failed to run')
    else:
        print('Job has not finished running yet...', flush=True)
        pass
        

def check_and_update_fitness(population, timeout=3600*2, loop_delay=100):
    # Set a timeout for each job
    while True:
        all_done = True
        for ind in population:
            gene_id = ind[0]
            # check if failed job
            if GLOBAL_DATA[gene_id]['sub_flag']==False:
                ind.fitness.values = (-float('inf'), float('inf')) # Max error
                GLOBAL_DATA[gene_id]['status'] == "completed"
            if ind.fitness.values == (-1234, 1234):  # If fitness not assigned
                # check for gene_id_model.txt file
                if GLOBAL_DATA[gene_id]['status'] == 'subbed file':
                    check4model2run(gene_id) # here we generate/sub model.txt file
                    # also we assign GLOBAL_DATA[gene_id]['status'] = 'running eval'
                if GLOBAL_DATA[gene_id]['status'] == 'running eval':
                    print(f'Checking for results for: {gene_id}')
                    check4results(gene_id) # here we look for the results file for gene_id
                    # also we assign GLOBAL_DATA[gene_id]['status'] = 'completed'
                if GLOBAL_DATA[gene_id]['status'] == "completed":
                    # Process results and assign fitness
                    fitness_tuple = GLOBAL_DATA[gene_id]['fitness']  # Implement this function
                    ind.fitness.values = fitness_tuple
                elif time.time() - GLOBAL_DATA[gene_id]['start_time'] > timeout:
                    print(f"Timeout for gene ID {gene_id}")
                    ind.fitness.values = (-float('inf'), float('inf')) 
                    GLOBAL_DATA[gene_id]['status'] = 'FAILED: TIMEOUT'
                else:
                    all_done = False  # Some jobs are still running
        if all_done:
            break  # All jobs are done or timed out
            print('\nEvalutated all genes\n')
        time.sleep(loop_delay)  # Wait some time before checking again
        
        
# Custom crossover function
def customCrossover(ind1, ind2):
    def combine_elements(ind1, ind2, cross_query_framework="queries/general/crossover_query.txt"):
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

        # Read model text for both individuals
        with open(os.path.join(out_dir, f'{gene_id_1}_model.txt'), 'r') as file:
            model_text_1 = file.read()
        with open(os.path.join(out_dir, f'{gene_id_2}_model.txt'), 'r') as file:
            model_text_2 = file.read()
        # Load crossover query framework
        with open(cross_query_framework, 'r') as file:
            cross_query_framework_text = file.read()
        # Generate the crossover query
        print(f'Mating: {gene_id_1} and {gene_id_2}')
        cross_query = cross_query_framework_text.format(model_text_1, model_text_2)
        # Generate a new gene ID for the offspring
        new_gene_id = generate_random_string(length=24)
        # Write the crossover query to a file
        query_path = os.path.join(out_dir, f'{new_gene_id}.txt')
        with open(query_path, 'w') as file:
            file.write(cross_query)
        # Create the bash file for the new job
        file_path = os.path.join(out_dir, f'{new_gene_id}.sh')
        successful_sub_flag, job_id = submit_bash(query_path, file_path, gpu='TeslaV100S-PCIE-32GB', 
                                                  fname='queries/py/query_instruct.py', max_seq_len=2048,
                                                  unique_id=new_gene_id, out_dir=out_dir)
        
        # Update global data for the new individual
        GLOBAL_DATA[new_gene_id] = {'sub_flag':successful_sub_flag, 'job_id':job_id, 
                                    'status':'subbed file', 'fitness':None, 'start_time':time.time()}
        
        if successful_sub_flag:
            print(f'Checking for job completion - cross: {job_id} for {new_gene_id}')
            job_done = check4job_completion(job_id)
            if job_done:
                print(f'Model files for {new_gene_id} are loaded')
            else:
                print(f'Error loading model files for {new_gene_id}!!!\n\n')
        
        failed_process = True if (successful_sub_flag is False) or (job_done is False) else False
        # Return the new gene ID
        return new_gene_id, failed_process
    
    global GLOBAL_DATA
    
    new_gene_id1, failed_process1 = combine_elements(ind1, ind2)
    new_gene_id2, failed_process2 = combine_elements(ind2, ind1)
    
    if failed_process1 is False:
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


def customMutation(individual, indpb, mutation_query_path="queries/general/mutation_query.txt"):
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
    # Load prompt tempate (This is a fixed template)
    with open(mutation_query_path, 'r') as file:
        mutation_query_text = file.read()
    # This should have been save from the last run
    with open(os.path.join(out_dir, f'{old_gene_id}_model.txt'), 'r') as file:
        model_text = file.read()
    # Update the old model with a mutation
    mutation_query_text = mutation_query_text.format(model_text)
    # Writing the string to the file
    query_path = os.path.join(out_dir, f'{new_gene_id}.txt')
    with open(query_path, 'w') as file:
        file.write(mutation_query_text)
    # Name of the sh bash file
    file_path = os.path.join(str(GENERATION), f'{new_gene_id}.sh')
    successful_sub_flag, job_id = submit_bash(query_path, file_path, gpu='TeslaV100S-PCIE-32GB', 
                                              fname='query_instruct.py', max_seq_len=2048,
                                              unique_id=new_gene_id, out_dir=out_dir)
    
    # Update the individual with the new gene ID
    # individual[0] = new_gene_id
    # Update the global data with the new task
    GLOBAL_DATA[new_gene_id] = {'sub_flag':successful_sub_flag, 'job_id':job_id, 
                                'status':'subbed file', 'fitness':None, 'start_time':time.time()}

    if successful_sub_flag:
        print(f'Checking for job completion Mutate: {job_id} for {new_gene_id}')
        job_done = check4job_completion(job_id)
        if job_done:
            print(f'Model files for {new_gene_id} are loaded')
        else:
            print(f'Error loading model files for {new_gene_id}')

    failed_process = True if (successful_sub_flag is False) or (job_done is False) else False
    
    if failed_process is False:
        individual[0] = new_gene_id
        individual = creator.Individual([new_gene_id])
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


# Define the problem
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Adjust weights as needed
creator.create("Individual", list, fitness=creator.FitnessMulti, file_id=None)
# Initialize the toolbox
toolbox = base.Toolbox()
# Register the modified function in the toolbox
toolbox.register("individual", create_individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalModel)
toolbox.register("mate", customCrossover)
toolbox.register("mutate", customMutation, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# --- Parameters --- #
GENERATION = 0
GLOBAL_DATA = {}
GLOBAL_DATA_HIST = {}
num_generations = 30  # Number of generations
start_population_size = 30  # Size of the population
population_size = 12
crossover_probability = 0.2  # Probability of mating two individuals
mutation_probability = 0.4   # Probability of mutating an individual
num_elites = 16

# Create a Hall of Fame
hof = HallOfFame(100)  # Argument is the number of individuals to keep
# Create an initial population
population = toolbox.population(n=start_population_size)
# Evaluate the entire population
for ind in population:
    # assign none to all so I can check them all at once
    ind.fitness.values = (-1234, 1234) 
check_and_update_fitness(population) # collect data

# Evolution
for gen in range(num_generations):
    print(f"Generation: {gen}", flush=True)
    # Select the next generation's parents
    elites = tools.selBest(population, num_elites)
    # Select the next generation's parents
    offspring = toolbox.select(population, population_size)
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    GLOBAL_DATA_HIST.update(GLOBAL_DATA.copy())
    # Remove individuals with placeholder fitness
    offspring = [ind for ind in offspring if ind.fitness.values != (-float('inf'), float('inf'))]
    
    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_probability:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            
    """
    Right now I cannot mate and 
    """
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
        # assign none to all so I can check them all at once
        ind.fitness.values = (-1234, 1234) 
    
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
    # Specify the filename for the pickle file
    filename = f'{gen}.pkl'
    # Open the file in write-binary ('wb') mode and save the dictionary
    with open(filename, 'wb') as file:
        pickle.dump(GLOBAL_DATA_HIST, file)
    print(f"GLOBAL_DATA_HIST saved as {filename}")

# At the end, the Hall of Fame contains the best individuals seen
print("Best ever individual:", hof[0])
# Print final results
print("-- End of Evolution --")
best_ind = tools.selBest(population, 1)[0]
print(f"Best Individual: {best_ind}")
print(f"Best Fitness: {best_ind.fitness.values}")