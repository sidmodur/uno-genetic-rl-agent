import environment as uno
import q_learning_agent as rlagent
import strategy_agent as sagent

# Agent parameters
agent_info = {"epsilon"  : .1,
              "gamma": .2,
              "alpha": 0, #decay
              "model": "../assets/models/q_v_rand/model",
              "learn": False
              }

# Load Q-Learning agent
q_v_rand = rlagent.QLearningAgent(agent_info)

# Run simulations
run = uno.tournament(iterations = 100,
                     agent1 = q_v_rand,
                     agent2 = None, #random strategy agent
                     comment = False)


print(run[0].count("q-learning"))