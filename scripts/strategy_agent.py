"""
Strategic agent uses domain-expertise to semi-intelligently calculate Q values,
with hyper-parameters that can be trained intelligently
"""
import pickle

load_from_file(path):
    with open(path, 'r') as file:
        return StrategicAgent(pickle.load(file))

class StrategicAgent:

    name = "Strategic"

    def __init__(self, parameters):
        self.params = parameters

    def step(self, player, open_card):
        pass

    def save_model(self, path):
        with open(path, 'w') as file:
            pickle.dump(self.params)
