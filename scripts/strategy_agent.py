"""
Strategic agent uses domain-expertise to semi-intelligently calculate Q values,
with hyper-parameters that can be trained intelligently
"""

import pickle


NUMBER = 1
COLOR = 1

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


    def rate_card(self, card, hand, discard_pile, N, C):
        x = 0 
        y = 0
        a = 0
        b = 0
        for h_card in hand - card:
            if card.color == h_card.color:
                x += 1
            if card.value == h_card.value:
                y += 1
        for d_card in discard_pile:
            if card.color == d_card.color:
                a += 1
            if card.value == d_card.value:
                b += 1
        # calculate score
        score = (C * x * a) + (N * y * b)
        return score

    """
    Gives each playable card a "gravity" score which approximates the ability of
    playing that card to move the game in the direction of the player. How much
    each card's attributes or hyper-attributes contribute to the gravity score
    will be an element of the hyperparameter list

    Takes in a player object, the open card, and returns the playable card object
    that has the highest gravity score.
    """
    def step(self, player, open_card, discard_pile):
        """
        Just a general idea for implementation:
        let x be the number of cards in hand with same color as the card in question
        let y be the number of cards in hand with same value as the card in question
        let a be the number of cards seen with same color as the card in question
        let b be the number of cards seen with same value as the card in question
        """
        """
        for each card in hand, calculate score
        score = (C_1 * x * a) + (C_2 * y * b)
        """

        max_score = 0
        best_card = None
        for card in player.hand:
            # calculate score
            score = self.rate_card(card, player.hand, discard_pile, NUMBER, COLOR)
            if max_score < score:
                max_score = score
                best_card = card
        
        return best_card
        """
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
