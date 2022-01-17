import environment as uno
import q_learning_agent as rlagent

# Agent parameters
agent_info = {"epsilon"  : .1,
              "gamma": .2,
              "alpha": 0, #decay
              "model": None,
              "learn": True
              }


# Run simulations
run = uno.tournament(iterations = 100,
                     agent1 = rlagent.QLearningAgent(agent_info),
                     agent2 = None, #random strategy agent
                     comment = False)

print(run[0].count("q-learning"))
