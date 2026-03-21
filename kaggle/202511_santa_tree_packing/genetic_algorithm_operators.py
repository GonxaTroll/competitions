import random
import numpy as np

def aux_NWOX(parent1,parent2,i,j): # Non-Wrapping Ordered Crossover (auxiliary function)
    child = [parent1[x] if i<=x<=j else None for x in range(len(parent1))] # Copying the block of one parent
    search_ind = 0
    for e2 in parent2: # Adding the rest of the elements in the order of the other parent
        if e2 not in child:
            while search_ind < len(child) and child[search_ind] is not None:
                search_ind += 1
            child[search_ind] = e2
    return child

def NWOX(parent1,parent2): # Non-Wrapping Ordered Crossover
    i = random.randint(0,len(parent1)-1)
    j = random.randint(0,len(parent1)-1)
    if j < i:
        i,j = j,i
    child1 = aux_NWOX(parent1,parent2,i,j)
    child2 = aux_NWOX(parent2,parent1,i,j)
    return [child1,child2]

def aux_CX(parent1,parent2): # Cycle Crossover (auxiliary function)
    parents = [parent1,parent2]
    N = len(parent1)
    child= [-1 for x in range(N)]
    end_cycle = True
    while sum(parent1) != sum(child):
        if end_cycle:
            current_index = child.index(-1)
            end_cycle = False
        else:
            posibles = [parent1[current_index],parent2[current_index]]
            choice = random.choice(posibles)
            selected_index_parent = posibles.index(choice)
            if choice in child:
                for i,p in enumerate(posibles):
                    if i != selected_index_parent:
                        if p in child:
                            end_cycle=True
                        else:
                            choice=p
                            selected_index_parent = i
                        break
            other_index_parent = abs(1-selected_index_parent)
            if choice not in child:
                child[current_index] = choice
                current_index = parents[other_index_parent].index(choice)
    return child

def CX(parent1,parent2): # Cycle Crossover
    return [aux_CX(parent1,parent2),aux_CX(parent2,parent1)]

def CX2_modded(parent1,parent2): # Cycle Crossover 2 corrected
    child1,child2 = [],[]
    original_parent1 = list(parent1)
    while len(child1)!=len(original_parent1):
        it = 0
        while True:
            if it==0:
                child1.append(parent2[0])
                it+=1
            else:
                child1.append(new_valueoff1)
            ref1p1 = parent1.index(child1[-1])
            ref2p1 = parent1.index(parent2[ref1p1])
            new_valueoff2 = parent2[ref2p1]
            child2.append(new_valueoff2)
            new_valueoff1 = parent2[parent1.index(new_valueoff2)]
            if new_valueoff1 in child1:
                break
        common = set(child1).intersection(child2)
        if len(common) != len(child1):
            child1 = child1 + [x for x in parent2 if x not in child1]
            child2 = child2 + [x for x in parent1 if x not in child2]
            break
        else:
            parent1 = [x for x in parent1 if x not in common]
            parent2 = [x for x in parent2 if x not in common]
    return [child1,child2]


def arithmetic_crossover(parent1_angles, parent2_angles, placed_trees = None):
    alpha = random.random()
    child1_angles, child2_angles = [], []
#   maintain placed_trees if provided
    if placed_trees is not None:
        for i in range(len(parent1_angles)):
            if i <= placed_trees:
                child1_angles.append(parent1_angles[i])
                child2_angles.append(parent2_angles[i])
            else:
                c1_angle = alpha * parent1_angles[i] + (1 - alpha) * parent2_angles[i]
                c2_angle = alpha * parent2_angles[i] + (1 - alpha) * parent1_angles[i]
                child1_angles.append(c1_angle)
                child2_angles.append(c2_angle)
        return [child1_angles, child2_angles]

    child1_angles = [alpha * a1 + (1 - alpha) * a2 for a1, a2 in zip(parent1_angles, parent2_angles)]
    child2_angles = [alpha * a2 + (1 - alpha) * a1 for a1, a2 in zip(parent1_angles, parent2_angles)]
    return [child1_angles, child2_angles]


def simple_swap(child): # Simple Swap (SS)
    i = random.randint(0,len(child)-1)
    j = random.randint(0,len(child)-1)
    child[i],child[j] = child[j],child[i]
    return child

def RSM(child): #Reverse Sequence Mutation: https://arxiv.org/ftp/arxiv/papers/1203/1203.3099.pdf#:~:text=In%20the%20reverse%20sequence%20mutation,covered%20in%20the%20previous%20operation.
    child = list(child) #returning a copy
    i = random.randint(0, len(child)-1)
    j = random.randint(0, len(child)-1)
    if j < i:
        i,j=j,i
    while i < j:
        child[i], child[j] = child[j], child[i]
        i += 1
        j -= 1
    return child

def aa_mutation(child, p_mutation_gene, placed_trees = None): # Angle Adjustment Mutation
    child = list(child)
    for i in range(len(child)):
        r = random.random()
        if r <= p_mutation_gene and (placed_trees is not None and i > placed_trees):
            mutation_angle = random.uniform(-30, 30)  # Mutate angle by up to ±30 degrees
            child[i] = (child[i] + mutation_angle) % 360  # Ensure angle stays within [0, 360)
    return child

def bitwise_mutation(child,p_mut=0.5): # Bitwise mutation
    child = list(child)
    for i in range(len(child)):
        r = random.random()
        if r <= p_mut:
            child[i] = abs(child[i]-1)
    return child


def ranked_wheel_selection(population, decoded_population, fitnesses, sp, parent_number):
    sorted_pop = sorted(population, key = lambda x: fitnesses[population.index(x)], reverse = True)
    sorted_decoded_pop = sorted(decoded_population, key = lambda x: fitnesses[decoded_population.index(x)], reverse = True)
    sorted_fitnesses = sorted(fitnesses, reverse = True)
    sorted_indices = list(range(len(population)))
    ranks = np.array(list(range(1,len(population)+1)))
    scaled_ranks = 2-sp + (2*(sp-1)*(ranks-1)/(ranks.size-1))
    selection_probs = scaled_ranks / np.sum(scaled_ranks)
    selected_indices = np.random.choice(sorted_indices,size=parent_number,p=selection_probs)
    new_pop = [sorted_pop[x] for x in selected_indices]
    new_sorted_decoded_pop = [sorted_decoded_pop[x] for x in selected_indices]
    new_fitnesses = [sorted_fitnesses[x] for x in selected_indices]
    return new_pop, new_sorted_decoded_pop, new_fitnesses


def elitist_replacement(population, decoded_population, population_fitnesses, children, decoded_children, children_fitnesses):
    new_population = population + children
    new_decoded_population = decoded_population + decoded_children
    new_fitnesses = population_fitnesses + children_fitnesses
    new_population = sorted(new_population, key = lambda x: new_fitnesses[new_population.index(x)])
    new_decoded_population = sorted(new_decoded_population, key = lambda x: new_fitnesses[new_decoded_population.index(x)])
    new_fitnesses = sorted(new_fitnesses)
    return new_population[0:len(population)], new_decoded_population[0:len(population)], new_fitnesses[0:len(population)]
