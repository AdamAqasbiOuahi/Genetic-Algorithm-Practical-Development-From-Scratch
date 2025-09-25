#!/usr/bin/env python
# coding: utf-8


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import math
import time
import itertools

file_png = 'A3_TEAM_A307\logo_gray_scale.png'
file_jpg = 'A3_TEAM_A307\logo_color.jpeg'
myself1 = 'A3_TEAM_A307\myself1.jpg'
myself2 = 'A3_TEAM_A307\myself2.png'
# .................................................................PART 1....................................................................

# (640, 366) jpeg = 234240 (factor 1.748)
# (640, 215) png = 137600 (factor 2.976)
# Load Image
def load_image (file, size):
# Load the file using PIL open and scale the image to a size no larger than the value size.
# Convert the PIL image into a NumPy array containing the pixels of the scaled image.
# The function must return a tuple containing the NumPy array and a Boolean variable
# denoting if the loaded image is a PNG file or not.

  # loading the image
  img = Image.open(file)
  # defining the shape without the last component (rgb or rbga)
  net_shape = np.shape(img)[:-1]

  # get the larger a lower shape component, and their respective index
  if net_shape[0]==net_shape[1]:
    max_axis, argmax_axis = net_shape[0],0
    min_axis, argmin_axis = net_shape[1],1
  else :
    max_axis, argmax_axis = np.max([net_shape[0], net_shape[1]]), np.argmax([net_shape[0], net_shape[1]])
    min_axis, argmin_axis = np.min([net_shape[0], net_shape[1]]), np.argmin([net_shape[0], net_shape[1]])
  # if the number that have been set as a maximum size (the parameter size) is lower than the actual maximum size
  # then resize the picture with the maximum size being the parameter
  if size < max_axis : #check if size is smaller than the maximum axis dimension
    ratio = max_axis/min_axis #calculate the ratio of the maximum to minimum axis dimensions
    new_size = [None, None] #start the new size list
    new_size[argmax_axis]=size #set size at the largest axis
    new_size[argmin_axis]=int(np.round(size/ratio)) #calculate and set the smallent dimension based on ratio
    new_size = tuple(new_size) #convert the list to a tuple to be able to put it in the resize function
    img = img.resize((new_size[1], new_size[0])) #resize the image using the new dimensions (the same way as shape but the other way around)

  img = np.array(img) #convert img to numpy array
  png = np.shape(img)[-1] == 4 #check if image has an alpha channel

  return img, png

def simplify_colors(image):
# Flatten the pixels of the image.
# Each component of each color of each pixel need to be either 0 or 255
# Values lower than or equal to 127 are converted to 0, higher than 127 to 255
  def flat_one(number):
    newnumber = np.round(number/255)*255 #normalize round and scale to 0 or 255
    return newnumber

  vectorized_flatting = np.vectorize(flat_one) #vectorize the flat one function for array use

  image = vectorized_flatting(image) #apply the vectorized function to the image array
  image

  return image

def compute_genotypes(image, png = True):
# Compute the probability of each genotype in the objective image
# In our case, it is the percentage of each color for pixels (remember PNG uses RGBA format, JPEG uses RGB)
# Hint: it is the probability of each color, not each component of each color
# Return the different genotypes and their probabilities
  reshapedimg = image.reshape(-1, 4 if png==True else 3) #reshape the image array to a 2D array with one color per row
  genotypes, freq = np.unique(reshapedimg, axis = 0,return_counts=True) #finding unique rows and their freqency in the reshaped image
  prob = freq/len(reshapedimg) #calculate probability of each genotype by dividing counts by the total number of pixels



  return genotypes, prob

def generate_random_image(image, genotypes, prob):
# Given the objective image, the different genotypes and their probability (prob),
# Generate a random image with the same size and the same phenotype probability as the objective image
  number_of_pixels = np.shape(image)[0]*np.shape(image)[1] #calculate the total number of pixels in the image

  # randomly select indices of genotypes based on their probability to fill the image
  pixels_chosen_index = np.random.choice(range(len(genotypes)), number_of_pixels, p=prob)
  pixels_chosen = genotypes[pixels_chosen_index] # get the actual genotype values using the randomly selected indices
  random_image = pixels_chosen.reshape(np.shape(image)) #reshape the flat array of genotypes back into the original image dimensions

  return random_image

def compute_search_space_size(file):
# Compute the size of the search space vs the size of the image
# comparing the original image stored in file and the simplified one
# Take into account the number of pixels and the number of colors to compute the total combinations
# As a suggestion, use a minimum size of 2 and a maximum of 600, step 16
# Return a tuple containing the number of combinations of the original image and the simplified one
  combinations = [] #new list to store combinations for original images
  simple_combinations = [] #store combinations for simplified images
  for number_of_pixels in range(10, 600, 10): #iterate over pixel counts
    imgarr, ispng = load_image(file, number_of_pixels) #load image with specific size

    unique_pixels = len(compute_genotypes(imgarr, ispng)[0]) #get unique color count
    combinations.append(math.log10(unique_pixels**(number_of_pixels**2))) #compute log10 of combinations

    imgarr = simplify_colors(imgarr) #simplify colors in image
    unique_pixels = len(compute_genotypes(imgarr, ispng)[0]) #get unique color count after simplification
    simple_combinations.append(math.log10(unique_pixels**(number_of_pixels**2))) #compute log10 of simple combinations


  return combinations, simple_combinations

def plot_search_space_size( combinations_png, combinations_jpg, simple_combinations_png, simple_combinations_jpg ):
# Plot the search space for the original images and the simplified images
# Verify how the search space drastically decreases when the color flattening is applied
# As suggested, for the x-axis, use a minimum size of 2 and a maximum of 600, step 16
  plt.plot(range(10, 600, 10), combinations_png)
  plt.plot(range(10, 600, 10), combinations_jpg)
  plt.plot(range(10, 600, 10), simple_combinations_png)
  plt.plot(range(10, 600, 10), simple_combinations_jpg)
  plt.legend(('png', 'jpg', 'simple png', 'simple jpg'))
  plt.xlabel('number of pixels (longuest side)')
  plt.ylabel('number of digits in the search space')
  plt.title(f'')
  plt.show()

def to_clock(seconds):
  seconds = round(seconds)
  m = seconds//60
  s = seconds - m*60
  return m , s

# ----- main for part 1 -----

beggin = time.time()
combinations_png, simple_combinations_png = compute_search_space_size(file_png)
combinations_jpg, simple_combinations_jpg = compute_search_space_size(file_jpg)
plot_search_space_size(combinations_png, combinations_jpg, simple_combinations_png, simple_combinations_jpg)
finish = time.time()
searchspacegraphtime = finish - beggin
print(f'time for search spaces graph : {to_clock(searchspacegraphtime)[0]}m and {to_clock(searchspacegraphtime)[1]}s')

# .................................................................PART 2....................................................................



def initial_population( goal, n_population, png=True):
# Generate the initial population using the objective image (goal)
# The initial population should be large enough to facilitate the exploration.
# Set the parameter n_population for the elements in this initial population
# Hint: use a reasonable value, not too big (long execution), not too small (solution not reachable)
# The initial population can be created at random or by using a heuristic
# For session 2, use random generation
  global genotypes
  global probs
  population = [generate_random_image(goal, genotypes, prob) for _ in range(n_population)]

  return population

def fitness_function(member, goal, png=True):
# Compute the fitness between the initial random image and the objective image
# Just count the different pixels between the two images

  # reshape in order to get the ordered list of pixels of each image
  reshapedmember, reshapedgoal = member.reshape(-1, 4 if png==True else 3), goal.reshape(-1, 4 if png==True else 3)
  # booleans indicates if each component of each pixel of the member image
  # is the same as the one of the corresponding component of the goal image
  booleans = reshapedmember == reshapedgoal
  # for each pixel, pixelbooleans indicates if the 3 components (or 4 in RGBA) of the member
  # are all the same as all the components of the goal (because we are camparing colors)
  pixelbooleans = booleans[:,0] & booleans[:,1] & booleans[:,2] & (booleans[:,3] if png else True)
  # 'fitness' will be the number of elements of pixelbooleans equal to True,
  # or, said differently, number of pixels that are the same for the member and goal image
  fitness = np.sum(pixelbooleans)
  return fitness

def selection(population, scores, k = 10):
# Selection of the parents from the population based on their fitness scores
# Default selection consists of k = 10 parents
# Selection can be implemented in several ways
# For session 2, use tournament
  # get index of max score from random k scores for len(population) times (this way we get as many parents as the previous population)
  selected_population_index = [scores.index(np.max(np.random.choice(scores, k))) for _ in range(len(population))]
  # use the indexes selectes to the the actual members that won the tournament
  selected_population = np.array(population)[selected_population_index]

  return selected_population

def crossover( p1, p2, r_cross):
# From the selected parents p1 and p2, produce the descendants c1 and c2
# Use r_cross parameter to decide if the crossover is produced or descendants are kept equal to parents
# Crossover can be in one single middle point, one single random point, multiple random points,
# multiple equally spaced points, alternating points, etc.
# Being a 2D matrix, the crossover can be either by column, by row or both
# For session 2, use one point random crossover
  if r_cross :
    cut_threshhold = round(np.shape(p1)[1]/2) #calculate crossover point at the middle column

    c1 = np.concatenate((p1[:, :cut_threshhold], p2[:, cut_threshhold:]), axis = 1) # the first child get first half from p1 second from p2
    c2 = np.concatenate((p2[:, :cut_threshhold], p1[:, cut_threshhold:]), axis = 1) # the second child get first half from p2 second from p1

    return c1, c2

  else :
    c1, c2 = p1, p2
    return c1, c2
  
def mutation( descendant, r_mut, genotypes, prob ):
# Mutate a descendant
# Mutation can be implemented in several ways: bti flip, swap, random, scramble, etc.
# Mutation should use only the possible genotypes with the given probability prob
# For session 2, use random mutation of each allele (pixel) with a probability r_mut

  muted_descendant = descendant.copy() #create a copy of the descendant to mutate
  for i in range(np.shape(descendant)[0]) : #iterate over rows
    for j in range(np.shape(descendant)[1]) : #iterate over the columns
      if rd.uniform(0,1) < r_mut : #check if mutation occurs based on mutation rate
        color_chosen = np.random.choice(range(len(genotypes)), 1, p = prob)[0] #choose a new color genotype based on given probabilities
        muted_descendant[i,j] = genotypes[color_chosen] #set new red value

  return muted_descendant

def replacement( population, descendants, r_replace ):
# Replacement of the population with the desdendants
# It can be implemented in several ways
# For session 2, just replace all old population with the new descendants (probability is 100%)

  return descendants

def genetic_algorithm( goal, n_iter, n_pop, r_cross, r_mut, r_replace, png=True ):
# Genetic algorithm should:
# 1. Generate the initial population
# 2. Start a loop with n_iter iterations
#    a. Inside the loop, evaluate the fitness of the population and store the best one
#    b. Make the selection of the parents
#    c. Crossover each couple of parents to generate new descendants
#    d. Mutate the descendants to create diversity
#    e. Replace the old population with the new one (descendants)
# 3. Return the best solution and the best fitness
  global mutation_time
  global crossover_time
  global selection_time

  global allscores
  global k

  pop = initial_population(goal, n_pop, png) #generate initial population


  for iter in range(n_iter):

    scores = [fitness_function(pop[i], goal, png) for i in range(len(pop))] #calculate fitness for each individual
    print(f'iteration : {iter+1} on {n_iter}') #print current best score
    print(f'best score : {np.max(scores)}, target : {fitness_function(goal, goal,png)}')
    print(f'mutation rate : {round(r_mut*100, 4)} percent \n')
    allscores.append(np.max(scores)) #store max score for this iteration

    selection_time1 = time.time() #start timing selection process
    selected = selection(pop, scores, k) #select parents based on scores

    selection_time2 = time.time() # end timing and update total selection time
    selection_time += selection_time2 - selection_time1

    crossover_time1 = time.time() #start timing crossover process

    if n_pop%2 == 0 : #perform crossover in pairs if population size is even
      crossed_selected = []
      for i in range(0, len(pop), 2): #crossover in pairs
        cross = crossover(selected[i], selected[i+1], r_cross)
        crossed_selected.append(cross[0]) #add first child to new population
        crossed_selected.append(cross[1]) #add second child to new population
      crossover_time2 = time.time() #end timing and update total crossover time
      crossover_time += crossover_time2 - crossover_time1
    else :
      crossed_selected = selected #use selected individuals directly if population is odd

    # mutate new individuals
    time1 = time.time() #start timing mutation process
    mutated_selected = [mutation(crossed_selected[i] , r_mut, *compute_genotypes(crossed_selected[i], png)) for i in range(len(crossed_selected))] #mutate new process
    time2 = time.time() #end timing and update total mutation time
    mutation_time += time2 - time1

    pop = replacement(pop, mutated_selected, r_replace) #replace old population with new one


  scores = [fitness_function(pop[i], goal, png) for i in range(len(pop))] #evaluate final population to find the best individual

  best_index = np.argmax(scores) #index of the best individual
  best = pop[best_index] #best individual
  best_eval = scores[best_index] #fitness of the best individual

  return best, best_eval

# ----- main for part 2 ------

# 2) Our part 2 model on file_png

# hyperparameters :
resolution = 100
n_iter = 2000
n_pop = 30
r_cross = True
r_mut = 0.1
r_replace = 1
k = n_pop

# initialise some variables
allscores = []
selection_time = 0
mutation_time = 0
crossover_time = 0

# get the goal image and other useful variables
goal, ispng = load_image(file_png, resolution)
goal = simplify_colors(goal)
goal_score = fitness_function(goal, goal, ispng)
genotypes, prob = compute_genotypes(goal, ispng)

# run the alrgorithm
start = time.time()
genetic_result, genetic_result_score = genetic_algorithm( goal, n_iter, n_pop, r_cross, r_mut/100, r_replace, ispng)
end = time.time()
total_time = end - start

print('\n\n\n')
print(f'total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
print(f'total selection time : {to_clock(selection_time)[0]}m and {to_clock(selection_time)[1]}s')
print(f'total mutation time : {to_clock(mutation_time)[0]}m and {to_clock(mutation_time)[1]}s')
print(f'total crossover time : {to_clock(crossover_time)[0]}m and {to_clock(crossover_time)[1]}s')

# plot the graph
plt.plot(range(1, len(allscores)+1), allscores)
plt.xlabel('iteration number')
plt.ylabel('best fitness score')
plt.title(f'Mutation rate {r_mut} percent\npopulation size : {n_pop}, k for selection : {k} \nFinal result : {genetic_result_score} on {goal_score}, total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
plt.ylim(allscores[0]*0.95, goal_score*1.05)
plt.xlim(1, n_iter+1)

plt.grid()
plt.axhline(goal_score, color='red')
plt.show()

# plot the images (goal and algo result)
plt.imshow(goal)
plt.show()
plt.imshow(genetic_result)
plt.show()


# .................................................................PART 3....................................................................



def selection_improved(population, scores, k = 10):
# Selection of the parents from the population based on their fitness scores
# Default selection consists of k = 10 parents
# Selection can be implemented in several ways
# For session 2, use tournament
  if k == len(population):
    selected_population = np.array([population[np.argmax(scores)] for _ in range(len(population))])
  else :
    # get index of max score from random k scores for len(population) times (this way we get as many parents as the previous population)
    selected_population_index = [scores.index(np.max(np.random.choice(scores, k))) for _ in range(len(population))]
    # use the indexes selectes to the the actual members that won the tournament
    selected_population = np.array(population)[selected_population_index]

  return selected_population

def mutation_improved( descendant, indexes_chosen, genotypes_chosen):
# Mutate a descendant
# Mutation can be implemented in several ways: bti flip, swap, random, scramble, etc.
# Mutation should use only the possible genotypes with the given probability prob
# For session 2, use random mutation of each allele (pixel) with a probability r_mut

  muted_descendant = descendant.copy()

  muted_descendant[indexes_chosen[0], indexes_chosen[1]] = genotypes_chosen

  return muted_descendant

def genetic_algorithm_improved1( goal, n_iter, n_pop, r_cross, r_mut_array, r_replace, png, breaking_condition):
# Genetic algorithm should:
# 1. Generate the initial population
# 2. Start a loop with n_iter iterations
#    a. Inside the loop, evaluate the fitness of the population and store the best one
#    b. Make the selection of the parents
#    c. Crossover each couple of parents to generate new descendants
#    d. Mutate the descendants to create diversity
#    e. Replace the old population with the new one (descendants)
# 3. Return the best solution and the best fitness
  global selection_time
  global mutation_time
  global crossover_time

  global allfinalscores
  global k_test

  global allscores
  global k
  global genotypes
  global prob

  pop = initial_population(goal, n_pop, png) #generate initial population
  n_pix = np.shape(goal)[0]*np.shape(goal)[1] #calculate the number of pixels in the goal image
  all_indexes = np.array(list(itertools.product(range(np.shape(goal)[0]), range(np.shape(goal)[1])))) #precompute all possible pixel indices

  for iter in range(n_iter):

    scores = [fitness_function(pop[i], goal, png) for i in range(len(pop))] #calculate fitness for each individual
    if k_test != True :
      print(f'iteration : {iter+1} on {n_iter}') #print current best score
      print(f'best score : {np.max(scores)}, target : {goal_score}')
      print(f'mutation rate : {round(r_mut_array[iter]*100, 4)} percent \n')
    allscores.append(np.max(scores))
    if iter > 2000 and breaking_condition : #check breaking condition to potentially terminate early
      if allscores[iter]-allscores[iter-2000] < 20 :
        break

    selection_time1 = time.time() #time the selection phase

    selected = selection_improved(pop, scores, k) #select parents based on scores

    selection_time2 = time.time()
    selection_time += selection_time2 - selection_time1



    crossover_time1 = time.time() #time the crossover phase

    if n_pop%2 == 0 : #perform crossover if population size is even
      crossed_selected = []
      for i in range(0, len(pop), 2): #crossover in pairs
        cross = crossover(selected[i], selected[i+1], r_cross)
        crossed_selected.append(cross[0]) #add first child to new population
        crossed_selected.append(cross[1]) #add second child to new population
    else :
      crossed_selected = selected

    crossover_time2 = time.time()
    crossover_time += crossover_time2 - crossover_time1

    # mutate new individuals
    time1 = time.time()  #time the mutation phase

    indexes_to_mute = np.random.choice(range(n_pix), round(r_mut_array[iter]*n_pix), replace=False) #calculate indices to mutate and the corresponding genotypes
    genotypes_indexes = genotypes[np.random.choice(range(len(genotypes)), round(r_mut_array[iter]*n_pix), p = prob)]

    mutated_selected = [mutation_improved(crossed_selected[i], all_indexes[(indexes_to_mute+(3*i))%n_pix].T, genotypes_indexes) for i in range(len(crossed_selected))] #perform improved mutation on selected individuals

    time2 = time.time()
    mutation_time += time2 - time1

    pop = replacement(pop, mutated_selected, r_replace) #replace old population with new one


  scores = [fitness_function(pop[i], goal, png) for i in range(len(pop))] #evaluate final population to find the best individual

  best_index = np.argmax(scores) #index of the best individual
  best = pop[best_index] #best individual
  best_eval = scores[best_index] #fitness of the best individual

  return best, best_eval

def genetic_algorithm_improved2( goal, n_iter, n_pop, r_cross, default_r_mut, distance_to_r_mut_ratio, exploitation_threshhold, r_replace, png, breaking_condition):
# Genetic algorithm should:
# 1. Generate the initial population
# 2. Start a loop with n_iter iterations
#    a. Inside the loop, evaluate the fitness of the population and store the best one
#    b. Make the selection of the parents
#    c. Crossover each couple of parents to generate new descendants
#    d. Mutate the descendants to create diversity
#    e. Replace the old population with the new one (descendants)
# 3. Return the best solution and the best fitness
  global selection_time
  global mutation_time
  global crossover_time

  global allfinalscores
  global allscores

  global k

  global genotypes
  global prob

  pop = initial_population(goal, n_pop, png) #generate initial population
  n_pix = np.shape(goal)[0]*np.shape(goal)[1] #calculate total number of pixels in the goal image
  all_indexes = np.array(list(itertools.product(range(np.shape(goal)[0]), range(np.shape(goal)[1])))) #precompute all possible pixel indices for efficiency

  for iter in range(n_iter):


    scores = [fitness_function(pop[i], goal, png) for i in range(len(pop))] #calculate fitness for each individual
    print(f'iteration : {iter+1} on {n_iter}') #print current best score
    print(f'best score : {np.max(scores)}, target : {goal_score}')
    allscores.append(np.max(scores))
    if iter > 2000 and breaking_condition : #early stopping condition based on score improvement
      if allscores[iter]-allscores[iter-2000] < 20 :
        break
    if iter > exploitation_threshhold : #adjust mutation rate based on performance relative to goal
      distance = 1-(np.max(scores)/goal_score)
      r_mut = distance/distance_to_r_mut_ratio
    else :
      r_mut = default_r_mut

    print(f'mutation rate : {round(r_mut*100, 4)} percent \n')

    selection_time1 = time.time() # time the selection process

    selected = selection_improved(pop, scores, k) #select parents based on scores

    selection_time2 = time.time()
    selection_time += selection_time2 - selection_time1


    crossover_time1 = time.time() #time the crossover process
    if n_pop%2 == 0 : #perform crossover in pairs if population size is even
      crossed_selected = []
      for i in range(0, len(pop), 2): #crossover in pairs
        cross = crossover(selected[i], selected[i+1], r_cross)
        crossed_selected.append(cross[0]) #add first child to new population
        crossed_selected.append(cross[1]) #add second child to new population
    else :
      crossed_selected = selected

    crossover_time2 = time.time()
    crossover_time += crossover_time2 - crossover_time1



    # mutate new individuals
    time1 = time.time()

    indexes_to_mute = np.random.choice(range(n_pix), round(r_mut*n_pix), replace=False) #calculate indices to mutate and corresponding genotypes
    genotypes_indexes = genotypes[np.random.choice(range(len(genotypes)), round(r_mut*n_pix), p = prob)]

    mutated_selected = [mutation_improved(crossed_selected[i], all_indexes[(indexes_to_mute+(3*i))%n_pix].T, genotypes_indexes) for i in range(len(crossed_selected))] #perform mutation on selected individuals

    time2 = time.time()
    mutation_time += time2 - time1

    pop = replacement(pop, mutated_selected, r_replace) #replace old population with new one


  scores = [fitness_function(pop[i], goal, png) for i in range(len(pop))] #evaluate final population to find the best individual

  best_index = np.argmax(scores) #index of the best individual
  best = pop[best_index] #best individual
  best_eval = scores[best_index] #fitness of the best individual

  return best, best_eval


# --------main for part 3--------

# 3) improved genetic algorithm for the same image as in part 2 (first imrovement)

# hyperparameters :
resolution = 100
n_iter = 3000
n_pop = 30
r_cross = True
r_mut = 0.1
r_replace = 1
k = n_pop
r_mut_start_percentage = 0.1
r_mut_final_percentage = 0.1
breaking_condition = False


# initialise some variables
allscores = []
selection_time = 0
mutation_time = 0
crossover_time = 0
r_mut_array = np.linspace(r_mut_start_percentage/100, r_mut_final_percentage/100, n_iter)
k_test = False

# get the goal image and other useful variables
goal, ispng = load_image(file_png, resolution)
goal = simplify_colors(goal)
goal_score = fitness_function(goal, goal, ispng)
genotypes, prob = compute_genotypes(goal, ispng)

# run the alrgorithm
start = time.time()
genetic_result, genetic_result_score = genetic_algorithm_improved1( goal, n_iter, n_pop, r_cross, r_mut_array, r_replace, ispng, breaking_condition)
end = time.time()
total_time = end - start

print('\n\n\n')
print(f'total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
print(f'total selection time : {to_clock(selection_time)[0]}m and {to_clock(selection_time)[1]}s')
print(f'total mutation time : {to_clock(mutation_time)[0]}m and {to_clock(mutation_time)[1]}s')
print(f'total crossover time : {to_clock(crossover_time)[0]}m and {to_clock(crossover_time)[1]}s')

# plot the graph
plt.plot(range(1, len(allscores)+1), allscores)
plt.xlabel('iteration number')
plt.ylabel('best fitness score')
plt.title(f'Mutation rate from {r_mut_array[0]*100} percent to {r_mut_array[-1]*100} percent \npopulation size : {n_pop}, k for selection : {k} \nFinal result : {genetic_result_score} on {goal_score}, total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
plt.ylim(allscores[0]*0.95, goal_score*1.05)
plt.xlim(1, n_iter+1)
plt.grid()
plt.axhline(goal_score, color='red')
plt.show()

# plot the images (goal and algo result)
plt.imshow(goal)
plt.show()
plt.imshow(genetic_result)
plt.show()


# 4) graph to find what is the best hyperparameter k

# hyperparameters :
resolution = 100
n_iter = 3000
n_pop = 30
r_cross = True
r_replace = 1
r_mut_start_percentage = 0.1
r_mut_final_percentage = 0.1

# initialise some variables
allscores = []
selection_time = 0
mutation_time = 0
crossover_time = 0
r_mut_array = np.linspace(r_mut_start_percentage/100, r_mut_final_percentage/100, n_iter)
allfinalscores = []
k_test = True
test = 0
k_array = np.linspace(1,30,15).astype(int)

# get the goal image and other useful variables
goal, ispng = load_image(file_png, resolution)
goal_score = fitness_function(goal, goal, ispng)
goal = simplify_colors(goal)
genotypes, prob = compute_genotypes(goal, ispng)

print(f'hyperparameter k test : {k_array}\n')

start = time.time()
k=1
for i in k_array:
  k = i
  allscores = []
  test += 1
  r_mut_array = np.linspace(r_mut_start_percentage/100, r_mut_final_percentage/100, n_iter)
  genetic_result_score = genetic_algorithm_improved1(goal, n_iter, n_pop, r_cross, r_mut_array, r_replace, ispng, False)[1]
  allfinalscores.append(genetic_result_score)
  print(f'\n\n\n\n\nTest number {test} on {len(k_array)}\nFor k = {k}\nFinal fitness score : {genetic_result_score} on {goal_score}\n\n\n\n\n\n')
end = time.time()
total_time = end - start

print(f'total k search time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(k_array, allfinalscores)
plt.xlabel('value of k')
plt.ylabel('final fitness score')
plt.title(f'Population size : {n_pop} \n mutation rate : {r_mut_array[0]*100} percent\nNumber of iteration : {n_iter}')
plt.ylim(np.min(allfinalscores)*0.95, goal_score*1.05)
# ax.set_yticks(np.arange(allscores[0]*0.95, goal_score*1.05, 100))
plt.grid()
plt.axhline(goal_score, color='red')
plt.show()



# 5) Our best model that approximates file_jpg

# hyperparameters :
resolution = 200
n_iter = 50000
n_pop = 30
r_cross = True
distance_to_r_mut_ratio = 1000
default_r_mut = 0.05
exploitation_threshhold = 4500
r_replace = 1
k = n_pop
breaking_condition = False

# initialise some variables
allscores = []
selection_time = 0
mutation_time = 0
crossover_time = 0
k_test = False

# get the goal image and other useful variables
goal, ispng = load_image(file_jpg, resolution)
goal = simplify_colors(goal)
goal_score = fitness_function(goal, goal, ispng)
genotypes, prob = compute_genotypes(goal, ispng)

# run the alrgorithm
start = time.time()
genetic_result, genetic_result_score = genetic_algorithm_improved2( goal, n_iter, n_pop, r_cross, default_r_mut/100, distance_to_r_mut_ratio, exploitation_threshhold, r_replace, ispng, breaking_condition)
end = time.time()
total_time = end - start

print('\n\n\n')
print(f'total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
print(f'total selection time : {to_clock(selection_time)[0]}m and {to_clock(selection_time)[1]}s')
print(f'total mutation time : {to_clock(mutation_time)[0]}m and {to_clock(mutation_time)[1]}s')
print(f'total crossover time : {to_clock(crossover_time)[0]}m and {to_clock(crossover_time)[1]}s')

# for genetic algorithm improved 2
plt.plot(range(1, len(allscores)+1), allscores)
plt.xlabel('iteration number')
plt.ylabel('best fitness score')
plt.title(f'Population size : {n_pop}, k for selection : {k}\nexploitation_threshhold = {exploitation_threshhold}, distance_to_r_mut_ratio : {distance_to_r_mut_ratio} \nFinal result : {genetic_result_score} on {goal_score}, Total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
plt.ylim(allscores[0]*0.95, goal_score*1.05)
plt.xlim(1, n_iter+1)

plt.grid()
plt.axhline(goal_score, color='red')
plt.show()

plt.imshow(goal)
plt.show()


plt.imshow(genetic_result)
plt.show()



# 6) Our best model that approximates file_png

# hyperparameters :
resolution = 200
n_iter = 50000
n_pop = 30
r_cross = True
distance_to_r_mut_ratio = 200
default_r_mut = 0.01
exploitation_threshhold = 2000
r_replace = 1
k = n_pop
breaking_condition = True

# initialise some variables
allscores = []
selection_time = 0
mutation_time = 0
crossover_time = 0
k_test = False

# get the goal image and other useful variables
goal, ispng = load_image(file_png, resolution)
goal = simplify_colors(goal)
goal_score = fitness_function(goal, goal, ispng)
genotypes, prob = compute_genotypes(goal, ispng)

# run the alrgorithm
start = time.time()
genetic_result, genetic_result_score = genetic_algorithm_improved2( goal, n_iter, n_pop, r_cross, default_r_mut/100, distance_to_r_mut_ratio, exploitation_threshhold, r_replace, ispng, breaking_condition)
end = time.time()
total_time = end - start

print('\n\n\n')
print(f'total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
print(f'total selection time : {to_clock(selection_time)[0]}m and {to_clock(selection_time)[1]}s')
print(f'total mutation time : {to_clock(mutation_time)[0]}m and {to_clock(mutation_time)[1]}s')
print(f'total crossover time : {to_clock(crossover_time)[0]}m and {to_clock(crossover_time)[1]}s')

# for genetic algorithm improved 2
plt.plot(range(1, len(allscores)+1), allscores)
plt.xlabel('iteration number')
plt.ylabel('best fitness score')
plt.title(f'Population size : {n_pop}, k for selection : {k}\nexploitation_threshhold = {exploitation_threshhold}, distance_to_r_mut_ratio : {distance_to_r_mut_ratio} \nFinal result : {genetic_result_score} on {goal_score}, Total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
plt.ylim(allscores[0]*0.95, goal_score*1.05)
plt.xlim(1, n_iter+1)

plt.grid()
plt.axhline(goal_score, color='red')
plt.show()

plt.imshow(goal)
plt.show()


plt.imshow(genetic_result)
plt.show()

# ----------------------------------------------------------------PART 4-----------------------------------------------------------------------------

def new_evolution_algorithm( goal, n_iter, r_mut, png, breaking_condition):
  
  global mutation_time

  global allfinalscores
  global allscores
  global k

  global genotypes
  global prob
  
  everyscore = []
  everyimage = []

  currentimage = generate_random_image(goal, genotypes, prob) #generate random image
  currentscore = fitness_function(currentimage, goal, png)
  everyimage.append(currentimage)
  everyscore.append(currentscore)

  n_pix = np.shape(goal)[0]*np.shape(goal)[1] #calculate total number of pixels in the goal image
  all_indexes = np.array(list(itertools.product(range(np.shape(goal)[0]), range(np.shape(goal)[1])))) #precompute all possible pixel indices for efficiency
  indexes_to_mute = np.linspace(0, n_pix, round(r_mut*n_pix)).astype(int)

  for iter in range(n_iter):
    if iter%1000 == 0:
      print(f'iteration : {iter} on {n_iter}') #print current best score
      print(f'current score : {everyscore[-1]}, target : {goal_score}')
      print(f'mutation rate : {round(r_mut*100, 4)} percent \n')

    if iter > 100000 and breaking_condition : #early stopping condition based on score improvement
      if allscores[-1]-allscores[-100000] < 10 :
        break

    # mutate
    time1 = time.time()

    chosen_genotypes = genotypes[np.random.choice(range(len(genotypes)), round(r_mut*n_pix), p = np.linspace(1/len(genotypes),1/len(genotypes),len(genotypes)))]
    
    current_image = mutation_improved(everyimage[-1], all_indexes[(indexes_to_mute + iter)%n_pix].T, chosen_genotypes)
    everyimage.append(current_image)
    everyscore.append(fitness_function(current_image, goal, png))
    allscores.append(fitness_function(current_image, goal, png))

    time2 = time.time()
    mutation_time += time2 - time1

    if everyscore[-1] > everyscore[-2]:
      pass
    else : 
      everyscore.pop()
      everyimage.pop()

  return everyimage, allscores


# ------- main ---------


# 7) Part 4 : new model approximation for file_jpg

# hyperparameters :

# file_png : 200
# file_jpg : 200
# myself1 : 150
# myself2 : 400
resolution = 200

n_iter = 600000

# file_jpg : 0.01
# myself1 : 0.007
r_mut = 0.01

# initialise some variables

selection_time = 0
mutation_time = 0
crossover_time = 0
allscores = []
breaking_condition=True

# get the goal image and other useful variables
goal, ispng = load_image(file_jpg, resolution)
goal = simplify_colors(goal)
goal_score = fitness_function(goal, goal, ispng)
genotypes, prob = compute_genotypes(goal, ispng)

# run the alrgorithm
start = time.time()
everyimage, allscores = new_evolution_algorithm( goal, n_iter, r_mut/100, ispng, breaking_condition)
end = time.time()
total_time = end - start

print('\n\n\n')
print(f'total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
print(f'total selection time : {to_clock(selection_time)[0]}m and {to_clock(selection_time)[1]}s')
print(f'total mutation time : {to_clock(mutation_time)[0]}m and {to_clock(mutation_time)[1]}s')
print(f'total crossover time : {to_clock(crossover_time)[0]}m and {to_clock(crossover_time)[1]}s')

# plot the graph
plt.plot(range(1, len(allscores)+1), allscores)
plt.xlabel('iteration number')
plt.ylabel('fitness score')
plt.title(f'Mutation rate {r_mut} percent\nFinal result : {allscores[-1]} on {goal_score}, total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
plt.ylim(allscores[0]*0.95, goal_score*1.05)
plt.xlim(1, n_iter+1)
plt.grid()
plt.axhline(goal_score, color='red')
plt.show()

# plot the images (goal and algo result)
plt.imshow(goal)
plt.show()
plt.imshow(everyimage[-1])
plt.show()




# 8) new model approximation for myself1.jpg

# hyperparameters :

# file_png : 200
# file_jpg : 200
# myself1 : 150
# myself2 : 400
resolution = 150

n_iter = 600000

# file_jpg : 0.01
# myself1 : 0.007
r_mut = 0.007

# initialise some variables

selection_time = 0
mutation_time = 0
crossover_time = 0
allscores = []
breaking_condition=True

# get the goal image and other useful variables
goal, ispng = load_image(myself1, resolution)
goal = simplify_colors(goal)
goal_score = fitness_function(goal, goal, ispng)
genotypes, prob = compute_genotypes(goal, ispng)

# run the alrgorithm
start = time.time()
everyimage, allscores = new_evolution_algorithm( goal, n_iter, r_mut/100, ispng, breaking_condition)
end = time.time()
total_time = end - start

print('\n\n\n')
print(f'total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
print(f'total selection time : {to_clock(selection_time)[0]}m and {to_clock(selection_time)[1]}s')
print(f'total mutation time : {to_clock(mutation_time)[0]}m and {to_clock(mutation_time)[1]}s')
print(f'total crossover time : {to_clock(crossover_time)[0]}m and {to_clock(crossover_time)[1]}s')

# plot the graph
plt.plot(range(1, len(allscores)+1), allscores)
plt.xlabel('iteration number')
plt.ylabel('fitness score')
plt.title(f'Mutation rate {r_mut} percent\nFinal result : {allscores[-1]} on {goal_score}, total time : {to_clock(total_time)[0]}m and {to_clock(total_time)[1]}s')
plt.ylim(allscores[0]*0.95, goal_score*1.05)
plt.xlim(1, n_iter+1)
plt.grid()
plt.axhline(goal_score, color='red')
plt.show()

# plot the images (goal and algo result)
plt.imshow(goal)
plt.show()
plt.imshow(everyimage[-1])
plt.show()