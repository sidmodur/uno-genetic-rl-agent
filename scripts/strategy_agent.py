"""
Strategic agent uses domain-expertise to semi-intelligently calculate Q values,
with hyper-parameters that can be trained intelligently
"""

import pickle

class StrategicAgent:

    name = "strategic"
    update = False

    def __init__(self, parameters=None, model=None):
        self.model = model
        if model is not None:
            with open(model, 'r') as file:
                self.params = pickle.load(file)
        elif parameters is not None:
            self.params = parameters
        else:
            raise Exception("Requires a list of parameters or the path to a pickled model")

    """
    Gives each playable card a "gravity" score which approximates the ability of
    playing that card to move the game in the direction of the player. How much
    each card's attributes or hyper-attributes contribute to the gravity score
    will be an element of the hyperparameter list

    Takes in a player object, the open card, and returns the playable card object
    that has the highest gravity score. 
    """
    def step(self, player, open_card):
        pass

    def save_model(self, path=None):
        if path == None:
            path = self.model
        if path != None:
            with open(path, 'w') as file:
                pickle.dump(self.params, file)
