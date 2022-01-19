import time
import q_learning_agent as qa
import genetic_search as genetic
import strategy_agent as sa
import environment as uno

q_agent_info = {
    "epsilon"  : .1,
    "gamma": .2,
    "alpha": 0, #alpha decay
    "model": None,
    "learn": True
}

q_agent = qa.QLearningAgent(q_agent_info)

run = uno.tournament(1000, q_agent, None, False)

print("done training")

strat_agent_info = {
    "model": None,
    "parameters": [1,1,1,1,1,1,1,1,1,1,1,1]
}

q_agent.learn = False

def struggle(agent):
    run = uno.tournament(50, agent, q_agent, False)
    wins = run[0].count(agent.name)
    return (wins, agent)

train_time = time.time()
search = genetic.GeneticSearch(sa.StrategicAgent(strat_agent_info), 1, 500, struggle)
end_time = time.time()
timer = end_time - train_time

print(search.winner.h)
print(f"Genetic search ran for: {timer}")