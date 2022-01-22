# CSCI 364 Final Project: Uno
### Comparing the effectiveness of intelligent and semi-intellegent agents in playing Uno

#### See `tests.py` for an example

## Running an Uno Tournament

The environment.py file contains the code to run an uno tournament, which is a set of uno games between two agents ran sequentially. The number of games in the tournament is determined by the user. A Uno tournament can be run by calling the `tournament(iterations, agent1, agent2, comment)` function in environment.py. It takes in 4 parameters: the number of games in the tournament, the two agents playing in the tournament, and a boolean indicating whether the game states should be printed out on screen. To play a random strategy agent, pass in `None` for one of the agents. The function returns a 3-tuple of a list of winners, a list of turns, and the time elapsed in seconds. 

Example:

```python
run = uno.tournament(iterations = 100,
                     agent1 = q_agent,
                     agent2 = None, #random strategy agent
                     comment = False)
```


## Initiating agent objects

Our Q-Learning agent is implemented in the q_learning_agent.py file, and our strategic agent is implemented in strategy_agent.py. Both objects are initiated using a parameter dictionary. The name of the agent is by default its type, but can be changed by changing the `name` field of the agent. This is necessary when playing two agents of the same kind against each other, since there would be no way of diffrentiating winners and losers.

### Q-Learning Agent.

The dictionary passed into the `__init__` function must contain fields: "epsilon", "gamma", "alpha", "model", and "learn". Epsilon, gamma, alpha are floats corresponding to their values in the Bellman equation. Note that setting alpha to 0 will tell the agent to use a decaying learning rate. Model contains a string of a filepath where a trained model is stored. Set to None if it is not wished to save or load the model. Learn is a boolean indicating whether the model should keep learning as it plays, useful to set to `False` if testing a already trained model.

Example:

``` python
agent_info = {"epsilon"  : .1,
              "gamma": .2,
              "alpha": 0, #decay
              "model": "../assets/models/q_v_rand/model",
              "learn": False
}
q_agent = q_learning_agent.QLearningAgent(agent_info)
```

### Strategic Agent

The dictionary passed into the `__init__` function must contain fields: "model" and "parameters". Model contains a string of a filepath where a trained model is stored. Parameter contains a list of parameters. If loading a previously saved model, this can be set to None.

Example:

```python
s_agent_info = {
    "model": "../assets/models/strat_unopt/model", 
    "parameters": None
    }
    
s_agent = strategy_agent.StrategicAgent(s_agent_info)
```

### A Note About Loading/Saving Models:

Passing in a path under the "models" key is possible even if there is not model saved there. For Q-Learning, it will initiate a fresh model, and for our strategy agent, it will use whatever parameters specified. These models can be saved to this file path by calling the `save_model` function. The `save_model` function takes in an optional file path, of where to save the model. If no parameter is specified, the model will be saved to the path specified under "model" in the initiation dictionary, overwriting any existing models in that location. If no path or model is specified nothing will be saved.

## Genetic algorithm:

To initiate a genetic algorithm, call the `__init__` function as described below:

```python

    def __init__(self, adam, generations, pop_size, struggle, carryover=50,
    mutation_coeff=.25, fitness=.25):
    
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
```