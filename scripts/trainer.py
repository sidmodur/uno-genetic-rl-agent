import environment as game
import strategy_agent as sagent
import q_learning_agent as qagent
import genetic_search as genetic

start_time = time.time()

"""
first train a q-learning agent against a random strategy agent
"""

train_time = time.time()
# Agent parameters
agent_info = {"epsilon"  : .1,
              "gamma": .2,
              "alpha": 0, #alpha decay
              "model": "../assets/q_v_rand/model",
              "learn": True
              }

q_v_rand = qagent.QLearningAgent(agent_info)

# Run simulations
run = uno.tournament(iterations = 10000,
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
# Agent parameters
agent_info = {"epsilon": .1,
              "gamma": .2,
              "alpha": 0, #alpha decay
              "model": "../assets/q_v_strat/model",
              "learn": True
              }

q_v_strat = qagent.QLearningAgent(agent_info)
unopt_strat = sagent.StrategicAgent({"model": "../assets/strat_unopt/model", "parameters": "model"})

# Run simulations
run = uno.tournament(iterations = 10000,
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
    run = uno.tournament(100, agent, None, False)
    return (run[0].count("strategic"), agent)

train_time = time.time()
search = genetic.GeneticSearch(unopt_strat, 1000, 1000, test_1)

end_time = time.time()
timer = end_time - train_time
print(f"time to optimize strategic agent against a random strategy: {timer}")

with open("../assets/ga_new_winners_1.txt", 'w') as file:
    file.writeline("Generations where the fittest individual changed:")
    for round in search.winner_change: file.writeline(str(round))

strat_opt_rand = search.winner
strat_opt_rand.save_model("../assets/models/strat_opt_rand/model")

"""
Now lets optimize the strategic agent against the q-learning agent
"""

q_v_strat.learn = False

def test_2(agent):
    run = uno.tournament(100, agent, q_v_strat, False)
    return (run[0].count("strategic"), agent)

train_time = time.time()
search = genetic.GeneticSearch(unopt_strat, 1000, 1000, test_2)

end_time = time.time()
timer = end_time - train_time
print(f"time to optimize strategic agent against a q-learning agent: {timer}")

with open("../assets/ga_new_winners_2.txt", 'w') as file:
    file.writeline("Generations where the fittest individual changed:")
    for round in search.winner_change: file.writeline(str(round))

strat_opt_q = search.winner
strat_opt_q.save_model("../assets/models/strat_opt_q/model")

end_time = time.time()
timer = end_time - start_time
print(f"The time to train all models: {timer}")
