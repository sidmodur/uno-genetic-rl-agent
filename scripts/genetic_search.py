"""
A genetic algorithm that solves the search problem of optimizing the hyperparameters
used by the strategy agent. Utilizes multi-threading for quicker execution
"""

import random
import strategy_agent as sa
import environment as uno

class GeneticSearch:

    """
    Produces an offspring based on the two parents. It does this by randomly
    selecting parameter values from a normal distribution centered at the mean
    of the two parent's parameters. This element of randomness repicates the
    chaos of random genetic mutation. To augment this process, the standard
    deviation for the normal distribution is given by the formula: for p = |param - mean|:
    p/{mutation coefficient}. This means that parents who are more genetically
    similar will produce offspring with greater genetic diversity.

    This is somewhat of a more cleaner process than in the real world,
    but definitely less fun.

    Returns a strategy agent with new paramters.
    """
    def reproduce(self, mother, father):
        genome = [0.0]*len(mother.h)
        for i in range(0, len(mother.h)):
            mu = float((mother.h[i] + father.h[i]) / 2)
            if mu == 0: sigma = .1
            sigma = abs(mother.h[i] - mu)
            sigma /= self.mutation_coeff
            genome[i] = random.normalvariate(mu, sigma)
        return sa.StrategicAgent({"model": None, "parameters": genome})

    """
    Selects the fittest individuals of the generation according to the specification,
    and mates them with a neighbor in terms of fitness score. Returns a list of
    individuals that constitute the next generation.
    """
    def regenerate(self, generation):
        new_gen = [None]*self.pop_size

        for i in range(0, self.carryover):
            new_gen[i] = generation[i][1]

        num_pairs = int(self.fitness * self.pop_size)
        mothers = generation[0:num_pairs:2] #evens
        fathers = generation[1:num_pairs:2] #odds
        num_pairs = int(num_pairs / 2)

        for i in range(self.carryover, self.pop_size):
            new_gen[i] = self.reproduce(mothers[i % num_pairs][1], fathers[i % num_pairs][1])

        return new_gen

    """
    Runs the genetic search algorithm by testing a generation for fitness, selecting
    the most fit, reproducing the generation with the most fit individuals' offspring,
    and repeating the cycle for however many generations specified.
    """
    def run_search(self):
        for i in range(0, self.generations):
            generation = sorted(map(self.struggle, self.population), key=lambda x: x[0], reverse=True)
            winner = generation[0][1]

            if winner != self.winner:
                self.winner_changed.append(i)
                self.winner = winner

            self.population = self.regenerate(generation)


    """
    Populates the search environment with the primordial generation, using adam
    as the template. Individuation happens by randomly selecting parameters from
    a normal distribution centered on adam's parameter value.
    """
    def genesis(self, adam):
        population = [None]*self.pop_size
        population[0] = adam

        for i in range(1, self.pop_size):
            genome = [0.0]*len(adam.h)
            for x in range(0, len(adam.h)):
                genome[x] = random.gauss(adam.h[x], adam.h[x])

            population[i] = sa.StrategicAgent({"model": None, "parameters": genome})

        return population

    """
    Initializes a new genetic search object.

    Parameters
    ----------------------------------------------------------------------------
    # adam: the first of its species of course
    # generations: how many generations to run
    # pop_size: how many individuals per generations
    # struggle: a function that determines an individual's fitness, should return a
      tuple (#wins, agent)
    # carry_over: specifies the number of the fittest individuals that
      should be carried over to the next generations.
    # mutation_coeff (0, 1): the value used in the calculation p + p/mutation_coefficient
      in the reproduce() method. (See mathod comment for details)
    # fitness (0, 1): the fraction of the population deemed "fit" to reproduce


    Object Fields
    ----------------------------------------------------------------------------
    # population: a list of individuals that constitute the last generation
    # winner: the most fit individual of the population
    # winner_changed: a list of rounds where the winner changed
    """
    def __init__(self, adam, generations, pop_size, struggle, carryover=50,
    mutation_coeff=.25, fitness=.25):
        self.generations = generations
        self.pop_size = pop_size
        self.struggle = struggle
        self.carryover = carryover
        self.mutation_coeff = mutation_coeff
        self.fitness = fitness
        self.population = self.genesis(adam)
        self.winner = None
        self.winner_changed = []
        self.run_search()
