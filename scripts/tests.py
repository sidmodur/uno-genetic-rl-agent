import environment as uno
import q_learning_agent as rlagent
import strategy_agent as sagent

# Q-Learning Agent parameters
q_agent_info = {"epsilon"  : .1,
              "gamma": .2,
              "alpha": 0, #decay
              "model": "../assets/models/q_v_rand/model",
              "learn": False
}

# Load Q-Learning agent
q_agent = rlagent.QLearningAgent(q_agent_info)

# Strategic Agent parameters
s_agent_info = {
    "model": "../assets/models/strat_unopt/model", 
    "parameters": None}

# Load Strategic Agent
s_agent = sagent.StrategicAgent(s_agent_info)

# Run simulations
run = uno.tournament(iterations = 100,
                     agent1 = q_agent,
                     agent2 = s_agent,
                     comment = False)


print(run[0].count("q-learning"))