import environment as uno
import q_learning_agent as rlagent
import strategy_agent as sagent

# Agent parameters
agent_info = {
"model": None,
"parameters": [1,1,1,1,1,1,1,1,1,1,1,1]
}


# Run simulations
run = uno.tournament(iterations = 100,
                     agent1 = sagent.StrategicAgent(agent_info),
                     agent2 = None, #random strategy agent
                     comment = False)

print(run[0].count("strategic"))
