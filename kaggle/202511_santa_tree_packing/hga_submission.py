import random
import time
import numpy as np
import pandas as pd
from shapely.strtree import STRtree
from shapely.ops import unary_union
from decimal import Decimal
from joblib import Parallel, delayed

from blf import blf, scale_factor
from genetic_algorithm_operators import arithmetic_crossover, aa_mutation, ranked_wheel_selection, elitist_replacement
from tree_utils import ChristmasTree, scale_factor, generate_submission_file

def generate_random_solution(n_trees = 10):
    return [np.random.uniform(0, 360) for _ in range(n_trees)]

def generate_random_population(population_size = 100, n_trees = 10):
    return [generate_random_solution(n_trees) for _ in range(1, population_size + 1)]

def get_fitness(solution_decoded_trees):
    polygons = [x.polygon for x in solution_decoded_trees]
    polygons_tree = STRtree(polygons)
    num_trees = len(solution_decoded_trees)

    # Checking for collisions
    limit = 100 * scale_factor
    for i, tree in enumerate(solution_decoded_trees):
        if tree.center_x < -limit or tree.center_x > limit or \
           tree.center_y < -limit or tree.center_y > limit:
            return np.inf
        poly = tree.polygon
        indices = polygons_tree.query(poly)
        for index in indices:
            if index == i:  # don't check against self
                continue
            if poly.intersects(polygons[index]) and not poly.touches(polygons[index]):
                return np.inf

    # Calculate score for the group
    bounds = unary_union(polygons).bounds
    # Use the largest edge of the bounding rectangle to make a square boulding box
    side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

    group_score = (Decimal(side_length_scaled) ** 2) / (scale_factor**2) / Decimal(num_trees)
    return float(group_score)


def adjust_solution(solution, placed_trees = None):
    decoded_trees = [ChristmasTree(angle=angle) for angle in solution]
    if placed_trees is None:
        decoded_trees = blf(decoded_trees)
    else:
        new_trees = decoded_trees[placed_trees:]
        old_trees = decoded_trees[:placed_trees]
        decoded_trees = blf(new_trees, old_trees)
    return decoded_trees

def get_population_to_save(population, new_population, fitnesses, new_population_fitnesses, num_to_save):
    saved_population, saved_population_fitnesses = [], []
    full_population = population + new_population
    full_fitnesses = fitnesses + new_population_fitnesses
    for i, ind in enumerate(full_population):
        if ind not in saved_population:
            saved_population.append(ind)
            saved_population_fitnesses.append(full_fitnesses[i])
        # if len(saved_population) == num_to_save:
        #     break
    saved_population = sorted(saved_population, key = lambda x: saved_population_fitnesses[saved_population.index(x)])
    saved_population_fitnesses = sorted(saved_population_fitnesses)
    return saved_population[0:num_to_save], saved_population_fitnesses[0:num_to_save]


def execute(n_trees = 10, placed_trees = None, intro_population=None, configuration=None):
    if configuration is None:
        configuration = ('NWOX', 0.9, 'InheritMask', 'SS', 0.4, 'DivisionSelect', 'RWS-sp-1.5', 'EL', 20)

    # cross_name = configuration[0]
    p_cross = configuration[1]
    # mut_name = configuration[3]
    p_mut = configuration[4]
    # mask_mut = configuration[5]
    sel_name = configuration[6]
    rep_name = configuration[7]
    pop_size = configuration[8]

    final_variables = ["best_fitness", "iterations", "final_time", "final_population"]
    iteration_data = []
    ini_time = time.time()
    iterations = 0
    generated_children = pop_size // 2

    if "RWS" in sel_name:
        selection = ranked_wheel_selection
    if rep_name=="EL":
        replacement = elitist_replacement

    if placed_trees is None or (placed_trees is not None and intro_population is None):
        population = generate_random_population(population_size = pop_size, n_trees=n_trees)
        decoded_population = Parallel(n_jobs=-1)(delayed(adjust_solution)(ind, placed_trees=placed_trees) for ind in population)
    else:
        if intro_population is not None:
            population = [ind + [np.random.uniform(0, 360) for _ in range(n_trees - len(ind))] for ind in intro_population]
            if len(population) < pop_size:
                additional_population = generate_random_population(population_size = pop_size - len(population), n_trees=n_trees)
                additional_decoded_population = Parallel(n_jobs=-1)(delayed(adjust_solution)(ind) for ind in additional_population)
            else:
                additional_population = []
                additional_decoded_population = []
            decoded_population = Parallel(n_jobs=-1)(delayed(adjust_solution)(ind, placed_trees) for ind in population)
            population += additional_population
            decoded_population += additional_decoded_population

    fitnesses = [get_fitness(x) for x in decoded_population]

    population_to_save, _ = get_population_to_save(population, [], fitnesses, [], pop_size)

    best_fitness_individual_iteration = min(fitnesses)
    best_individual_iteration = population[fitnesses.index(best_fitness_individual_iteration)]
    iteration_data.append([best_fitness_individual_iteration,0,0,"pop"])
    best_last_fitness = best_fitness_individual_iteration
    repeated_iterations = 0

    while iterations < 200:
        # print(f"Iteration {iterations}, Best fitness: {best_fitness_individual_iteration}")
        ini_iteration_time = time.time()
        parents, _, _ = selection(
            population, decoded_population, fitnesses, 1.5, generated_children
        )

        random_indices = list(range(len(parents)))
        random.shuffle(random_indices)
        parents = [parents[i] for i in random_indices]

        all_children = []
        for i in range(0, generated_children - 1, 2):
            parent_1, parent_2 = parents[i], parents[i+1]
            r = random.random()
            if r <= p_cross:
                children = arithmetic_crossover(parent_1, parent_2, placed_trees=placed_trees)
            else:
                children = [parent_1, parent_2]

            mutated_children = []
            for i in range(len(children)):
                mutated_child = children[i]
                r = random.random()
                if r <= p_mut:
                    # mutated_child = mutation(mutated_child)
                    # mutated_child = RSM(mutated_child)
                    mutated_child = aa_mutation(mutated_child, p_mutation_gene=0.25, placed_trees=placed_trees)
                mutated_children.append(mutated_child)

            for child in mutated_children: # children + mutated_children:
                all_children.append(child)

        # decoded_children = [adjust_solution(ind) for ind in all_children]
        decoded_children = Parallel(n_jobs=-1)(delayed(adjust_solution)(ind, placed_trees=placed_trees) for ind in all_children)
        children_fitnesses = [get_fitness(x) for x in decoded_children]

        population_to_save, _ = get_population_to_save(population, children, fitnesses, children_fitnesses, pop_size)

        population, decoded_population, fitnesses = elitist_replacement(
            population, decoded_population, fitnesses, children, decoded_children, children_fitnesses
            )

        best_individual_iteration = population[0]
        best_individual_iteration_decoded = decoded_population[0]
        best_fitness_individual_iteration = fitnesses[0]
        iterations+=1
        end_iteration_time = time.time()-ini_iteration_time
        iteration_data.append([best_fitness_individual_iteration, iterations, end_iteration_time, "pop"])

        if best_fitness_individual_iteration >= best_last_fitness:
            repeated_iterations += 1
        else:
            repeated_iterations = 0
        best_last_fitness = best_fitness_individual_iteration

        if iterations == 100 or repeated_iterations == 20:
            final_time = time.time()-ini_time
            result_data = [[best_fitness_individual_iteration, iterations, final_time, population]]
            result_data = pd.DataFrame(result_data,columns=final_variables)
            iteration_data = pd.DataFrame(iteration_data,columns=final_variables)
            iteration_data.drop(columns=["final_population"],inplace=True)
            return best_individual_iteration, best_individual_iteration_decoded, result_data, iteration_data, population_to_save


if __name__ == "__main__":
    try:
        data = pd.read_csv("submission_hga_partitioned_incremental.csv")
        initial_trees = int(data["id"].str.split("_", expand=True)[0].astype(int).max())
    except:
        initial_trees = 1

    results = {}
    results_fitness = []
    for i in range(initial_trees, 201):
        t1 = time.time()
        print(f"Execution {i}")
        if i == 1:
            placed_trees, intro_population = None, None
        else:
            placed_trees = i - 1

        best_individual, best_individual_decoded, result_data, iteration_data, intro_population =\
            execute(n_trees = i, placed_trees = placed_trees, intro_population = intro_population)
        results[i] = best_individual_decoded
        results_fitness.append(result_data["best_fitness"][0])
        t2 = time.time()
        print(f"Execution {i} finished in {round(t2 - t1, 2)} seconds with fitness {result_data['best_fitness'][0]}")

        submission_df = generate_submission_file(results)
        submission_df.to_csv("submission_hga_partitioned_incremental.csv", index=False)
