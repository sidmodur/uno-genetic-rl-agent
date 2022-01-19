"""
Strategic agent uses domain-expertise to semi-intelligently calculate Q values,
with hyper-parameters that can be trained intelligently
"""

import pickle

class StrategicAgent:

    name = "strategic"
    update = False

    """
    Initializes the agent from a dictionary containing either a path to the model,
    or a list of hyperparameters. (Dictionary must contain both fields, with one set to None)

    There are 12 hyperparameters, listed by index:

    #0 The relative importance of the number of cards in hand with a certain color
    and the number of cards seen so far with that color.
    #1 The relative importance of the number of cards in hand with a certain value
    and the number of cards seen so far with that value.
    #2,3 hyperparameters for skip cards (see comment for getSpecWeight())
    #4,5 hyperparameters for reverse cards (see comment for getSpecWeight())
    #6,7 hyperparameters for +2 cards (see comment for getSpecWeight())
    #8,9 hyperparameters for +4 wild cards (see comment for getSpecWeight())
    #10,11 hyperparameters for normal wild cards (see comment for getSpecWeight())
    """
    def __init__(self, agent_init_info):
        self.model = agent_init_info["model"]
        self.h = agent_init_info["parameters"]
        self.color_count = dict()
        self.value_count = dict()

        if self.model is not None:
            try:
                with open(self.model) as file:
                    self.h = pickle.load(file)
            except OSError:
                print(f'no model found, will use specified parameters')

        if self.h is None:
            raise Exception("No model was found and no parameters were specified")

    """
    Returns the "bonus" portion of the gravity score which special cards get
    according to this formula: bonus = l + r*n, where:

    l is a hyper-parameter capturing the inherent value of a given special card
    relative to other cards,
    r is a hyper-parameter capturing the when in the game it is most advantagious
    to play the card,
    and n is how many cards have been played in the game so far
    """
    def getSpecWeight(self, card):
        weights = {
            "SKI": self.h[2] + (self.h[3]*self.card_count),
            "REV": self.h[4] + (self.h[5]*self.card_count),
            "PL2": self.h[6] + (self.h[7]*self.card_count),
            "PL4": self.h[8] + (self.h[9]*self.card_count),
            "COL": self.h[10] + (self.h[11]*self.card_count),
        }

        return weights[card.value]

    """
    Rates cards according to the formula: score = (C_1 * x * a) + (C_2 * y * b)

    where x is the number of cards in hand with same color as the card in question
    where y is the number of cards in hand with same value as the card in question
    where a is the number of cards seen with same color as the card in question
    where b is the number of cards seen with same value as the card in question
    where C_1, C_2 are hyperparameters capturing the importance of these components

    special cards also have a component of their score determined outside this
    function
    """
    def getGravity(self, card, player):
        x, y, a, b = 0, 0, 1, 1

        if card.value in ["PL4", "COL"]:
            card.color = player.choose_color()

        for h_card in player.hand:
            if card.color == h_card.color:
                x += 1
            if card.value == h_card.value:
                y += 1

        if card.color in self.color_count:
            a += self.color_count[card.color]

        if card.value in self.value_count:
            b += self.value_count[card.value]

        # calculate score
        score = (self.h[0]*x*a) + (self.h[1]*y*b)

        if card.value in ["SKI", "REV", "PL2", "PL4", "COL"]:
            score += self.getSpecWeight(card)

        return score

    """
    Gives each playable card a "gravity" score which approximates the ability of
    playing that card to move the game in the direction of the player. How much
    each card's attributes or hyper-attributes contribute to the gravity score
    will be an element of the hyperparameter list

    Takes in a player object, the open card, and returns the playable card object
    that has the highest gravity score.
    """
    def step(self, player, open_card):
        # Count open card
        if open_card.color in self.color_count:
            self.color_count[open_card.color] += 1
        else:
            self.color_count.update({open_card.color: 1})

        if open_card.value in self.value_count:
            self.value_count[open_card.value] += 1
        else:
            self.value_count.update({open_card.value: 1})

        self.card_count += 1
        

        max_grav = self.getGravity(player.hand_play[0], player)
        best_card = player.hand_play[0]
        for i in range(1, len(player.hand_play)):
            # calculate score
            gravity = self.getGravity(player.hand_play[i], player)
            if max_grav < gravity:
                max_grav = gravity
                best_card = player.hand_play[i]

        return best_card


    """
    Resets domain knowledge after each game
    """
    def reset(self):
        self.color_count = dict()
        self.value_count = dict()
        self.card_count = 0

    """
    Saves the model by pickling the list of hyperparameters
    """
    def save_model(self, path=None):
        if path == None:
            path = self.model
        if path != None:
            with open(path, 'w') as file:
                pickle.dump(self.params, file)
