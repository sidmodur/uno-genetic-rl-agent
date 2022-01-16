"""
A genetic algorithm that solves the search problem of optimizing the hyperparameters
used by the strategy agent. Utilizes multi-threading for quicker execution
"""

from multiprocessing import Pool
import math
import random
import strategy_agent as sa
import environment as uno

class GeneticSearch

    # "pads out" the new generation with the fittest members of the old one,
    # and ensuring that a minimum number of individuals are carried forth
    # to the new generation
    def pad_generation(self, gen, new_gen, start):
        start = min(start, len(new_gen) - self.min_carry_over)
        for i in range(0, self.pop_size - start):
            new_gen[start + i] = gen[i][1]

        return new_gen

    """
    Produces an offspring based on the two parents. It does this by randomly
    selecting parameter values from a normal distribution centered at the mean
    of the two parent's parameters. This element of randomness repicates the
    chaos of random genetic mutation. To augment this process, the standard
    deviation for the normal distribution is given by the formula: for p = |param - mean|:
    p + p/mutation_coefficient. This means that parents who are more genetically
    similar will produce offspring with greater genetic diversity.

    This is somewhat of a more family-friendly process than in the real world,
    but definitely less fun.

    Returns a strategy agent with new paramters.
    """
    def reproduce(self, mother, father):
        genome = []
        for i in range(0, len(mother.params)):
            mu = (mother.params[i] + father.params[i]) / 2
            if mu == 0: sigma = .1
            sigma = abs(mother.params[i] - mu)
            sigma += (sigma / self.mutation_coeff)
            genome[i] = random.normalvariate(mu, sigma)
        return sa.StrategicAgent(parameters=genome)

    """
    Selects the fittest individuals of the generation according to the specification,
    Returns a 2 lists of parents with equal lengths, and the length of the lists
    """
    def applySelection(self, generation):
        num_fit = math.floor(self.fitness * self.pop_size)
        if num_fit % 2 != 0: num_fit -= 1
        survivors = generation[0:unit_size]
        return survivors[0::2], survivors[1::2], unit_size/2

    """
    Runs the genetic search algorithm by testing a generation for fitness, selecting
    the most fit, reproducing the generation with the most fit individuals' offspring,
    and repeating the cycle for however many generations specified.
    """
    def run_search(self)
        for i in range(1, self.generations):
            with Pool.pool() as p:
                generation = p.map(self.struggle, self.population)
                generation.sort(reverse=True)
                winner = generation[0][1]

                if winner != self.winner:
                    self.winner_change += 1
                    self.winner = winner

                newGen = [None]*self.pop_size
                mothers, fathers, unit_size = self.applySelection(generation)

                start = 0
                end = unit_size
                while end < self.pop_size:
                    newGen[start:end] = p.map(self.reproduce, mothers, fathers)
                    start = end
                    end = start + unit_size

                self.population = self.pad_generation(newGen, start)

    """
    Populates the search environment with the primordial generation, using adam
    as the template. Individuation happens by randomly selecting parameters from
    a normal distribution centered on adam's parameter value.
    """
    def genesis(self, adam):
        population = [None]*self.pop_size
        population[0] = adam

        for i in range(1, self.pop_size):
            genome = []
            for x in range(0, len(adam.params)):
                genome[x] = random.gauss(adam[x], self.mutation_coeff * adam[x])

            population[i] = sa.StrategicAgent(parameters=genome)

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
    # min_carry_over: specifies the minimum number of the fittest individuals that
      should be carried over to the next generations. (Actual number may be higher, never lower)
    # mutation_coeff (0, 1): the value used in the calculation p + p/mutation_coefficient
      in the reproduce() method. (See mathod comment for details)
    # fitness (0, 1): the fraction of the population deemed "fit" to reproduce


    Object Fields
    ----------------------------------------------------------------------------
    # population: a list of individuals that constitute the last generation
    # winner: the most fit individual of the population
    # winner_changed: the number of times the winner changed while the search ran
    """
    def __init__(self, adam, generations, pop_size, struggle, min_carry_over=10,
    mutation_coeff=.25, fitness=.25):
        self.generations = generations
        self.pop_size = pop_size
        self.struggle = struggle
        self.carry_over = carry_over
        self.mutation_coeff = mutation_coeff
        self.fitness = fitness
        self.population = self.genesis(adam)
        self.winner = None
        self.winner_changed = 0
        self.run_search(adam)
