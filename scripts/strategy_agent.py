"""
Strategic agent uses domain-expertise to semi-intelligently calculate Q values,
with hyper-parameters that can be trained intelligently
"""

import pickle

class StrategicAgent:

    name = "strategic"
    update = False

    def __init__(self, agent_init_info):
        self.model = agent_init_info["model"]
        self.params = agent_init_info["parameters"]
        self.color_count = dict()
        self.card_count = dict()

        if self.model is not None:
            try:
                with open(self.model) as file:
                    self.params = pickle.load(file)
            except OSError:
                print(f'no model found, will use specified parameters')

        if self.params is None:
            raise Exception("No model was found and no parameters were specified")

    """
    Gives each playable card a "gravity" score which approximates the ability of
    playing that card to move the game in the direction of the player. How much
    each card's attributes or hyper-attributes contribute to the gravity score
    will be an element of the hyperparameter list

    Takes in a player object, the open card, and returns the playable card object
    that has the highest gravity score.
    """
    def step(self, player, open_card):
        """
        Just a general idea for implementation:
        let x be the number of cards in hand with same color as the card in question
        let y be the number of cards in hand with same value as the card in question
        let a be the number of cards seen with same color as the card in question
        let b be the number of cards seen with same value as the card in question

        for each card in hand, calculate score
        score = (C_1 * x * a) + (C_2 * y * b)

        Special cards will have different score calculations, but will each have
        a constant term that is added to the score which is a hyperparameter
        (think of this as the inherent value of a special card)
        """
        pass

    """
    Resets domain knowledge after each game
    """
    def reset(self):
        self.color_count = dict()
        self.card_count = dict()

    def save_model(self, path=None):
        if path == None:
            path = self.model
        if path != None:
            with open(path, 'w') as file:
                pickle.dump(self.params, file)
