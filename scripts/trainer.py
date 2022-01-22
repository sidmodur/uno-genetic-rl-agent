"""
A script to train our various models, keeps track of the time to train, which
is printed out.
"""

import environment as uno
import strategy_agent as sagent
import q_learning_agent as qagent
import genetic_search as genetic
import time

start_time = time.time()

"""
first train a q-learning agent against a random strategy agent
"""

train_time = time.time()
# Agent parameters
agent_info = {"epsilon"  : .1,
              "gamma": .1,
              "alpha": 0, #alpha decay
              "model": "../assets/models/q_v_rand/model",
              "learn": True
              }

q_v_rand = qagent.QLearningAgent(agent_info)

# Run simulations
run = uno.tournament(iterations = 100000,
                     agent1 = q_v_rand,
                     agent2 = None, #random strategy agent
                     comment = False)

end_time = time.time()
timer = end_time - train_time
print(f"time to train q_v_rand: {timer}")
q_v_rand.save_model()



"""
Now lets train the q-learning agent against the unoptimized strategy agent
"""

train_time = time.time()
print(f"training started at: {train_time}")
# Agent parameters
agent_info = {"epsilon": .1,
              "gamma": .1,
              "alpha": 0, #alpha decay
              "model": "../assets/models/q_v_strat/model",
              "learn": True
              }

q_v_strat = qagent.QLearningAgent(agent_info)
unopt_strat = sagent.StrategicAgent({"model": "../assets/models/strat_unopt/model", "parameters": "models"})

# Run simulations
run = uno.tournament(iterations = 100000,
                     agent1 = q_v_strat,
                     agent2 = unopt_strat,
                     comment = False)

end_time = time.time()
timer = end_time - train_time
print(f"time to train q_v_strat: {timer}")
q_v_strat.save_model()


"""
Now lets optimize the strategic agent against the random agent
"""

def test_1(agent):
    run = uno.tournament(50, agent, None, False)
    wins = run[0].count("strategic")
    return (wins, agent)

train_time = time.time()
search = genetic.GeneticSearch(unopt_strat, 500, 500, test_1)

end_time = time.time()
timer = end_time - train_time
print(f"time to optimize strategic agent against a random strategy: {timer}")

with open("../assets/models/ga_new_winners_1.txt", 'w') as file:
    file.writelines("Generations where the fittest individual changed:")
    for round in search.winner_changed: file.writelines(str(round))

strat_opt_rand = search.winner
strat_opt_rand.save_model("../assets/models/strat_opt_rand/model")