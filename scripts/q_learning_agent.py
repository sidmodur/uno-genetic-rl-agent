# 1. Libraries
# -------------------------------------------------------------------------

# Custom libraries
import state_action_reward as sar

# Public libraries
import pandas as pd
import numpy as np
import random


# 2. Q-Learning
# -------------------------------------------------------------------------

class QLearningAgent(object):

    name = "q-learning"
    update = True

    def __init__(self, agent_init_info):
        """
        Initializes the agent to get parameters and import/create q-tables.
        Required parameters: agent_init_info as dict
        """

        # (1) Store the parameters provided in agent_init_info
        self.states      = sar.states()
        self.actions     = sar.actions()
        self.prev_state  = 0
        self.prev_action = 0

        self.epsilon     = agent_init_info["epsilon"]
        self.gamma       = agent_init_info["gamma"]
        self.alpha       = agent_init_info["alpha"]
        self.model       = agent_init_info["model"]
        self.learn       = agent_init_info["learn"]
        self.R           = sar.rewards(self.states, self.actions)


        # (2) Create Q-table that stores action-value estimates, initialized at zero
        if self.model == None:
            self.q = pd.DataFrame(data    = np.zeros((len(self.states), len(self.actions))),
                                  columns = self.actions,
                                  index   = self.states)

            self.visit = self.q.copy()


        # (3) Import already existing Q-values and visits table if possible
        else:
            try:
                self.q            = pd.read_csv(self.model + "-q.csv", sep = ";", index_col = "Unnamed: 0")
                self.q.index      = self.q.index.map(lambda x: eval(x))
                self.q["IDX"]     = self.q.index
                self.q            = self.q.set_index("IDX", drop = True)
                self.q.index.name = None

                self.visit            = pd.read_csv(self.model + "-visits.csv", sep = ";", index_col = "Unnamed: 0")
                self.visit.index      = self.visit.index.map(lambda x: eval(x))
                self.visit["IDX"]     = self.visit.index
                self.visit            = self.visit.set_index("IDX", drop = True)
                self.visit.index.name = None

            # (3a) Create empty q-tables if file is not found
            except:
                print ("Existing model could not be found. New model is being created.")
                self.q = pd.DataFrame(data    = np.zeros((len(self.states), len(self.actions))),
                                      columns = self.actions,
                                      index   = self.states)

                self.visit = self.q.copy()

    def play_card(self, action, player, card_open):

        if self.learn: self.update(player.state, action)

        # Selected action searches corresponding card
        # (1) Playing wild card
        if player.action in ["COL","PL4"]:
            for card in player.hand:
                if card.value == player.action:
                    break

        # (2) Playing normal card with different color
        elif (player.action in ["RED","GRE","BLU", "YEL"]) and (player.action != card_open.color):
            for card in player.hand:
                if (card.color == player.action) and (card.value == card_open.value):
                    break

        # (3) Playing normal card with same color
        elif (player.action in ["RED","GRE","BLU", "YEL"]) and (player.action == card_open.color):
            for card in player.hand:
                if (card.color == player.action) and (card.value in range(0,10)):
                    break

        # (4) Playing special card with same color
        elif (player.action not in ["RED","GRE","BLU", "YEL"]) and (player.action != card_open.value):
            for card in player.hand:
                if (card.color == card_open.color) and (card.value == player.action):
                    break

        # (5) Playing special card with different color
        else:
            for card in player.hand:
                if card.value == player.action:
                    break

        return card

    def step(self, player, open_card):
        """
        Choose the optimal next action according to the followed policy.
        Required parameters:
            - state_dict as dict
            - actions_dict as dict
        """

        player.identify_state(open_card)
        player.identify_action()

        # (1) Transform state dictionary into tuple
        state = [i for i in player.state.values()]
        state = tuple(state)

        # (2) Choose action using epsilon greedy
        # (2a) Random action
        if random.random() < self.epsilon:

            actions_possible = [key for key,val in player.actions.items() if val != 0]
            action = random.choice(actions_possible)

        # (2b) Greedy action
        else:
            actions_possible = [key for key,val in player.actions.items() if val != 0]
            random.shuffle(actions_possible)
            val_max = 0

            for i in actions_possible:
                val = self.q.loc[[state],i][0]
                if val >= val_max:
                    val_max = val
                    action = i

        return self.play_card(action, player, open_card)


    def update(self, state_dict, action):
        """
        Updating Q-values according to Belman equation
        Required parameters:
            - state_dict as dict
            - action as str
        """
        state = [i for i in state_dict.values()]
        state = tuple(state)

        # (1) Set prev_state unless first turn
        if self.prev_state != 0:
            prev_q = self.q.loc[[self.prev_state], self.prev_action][0]
            this_q = self.q.loc[[state], action][0]
            reward = self.R.loc[[state], action][0]

            if self.alpha == 0:
                alpha = self.visit.loc[[self.prev_state], self.prev_action][0]
                alpha = 1.0/alpha if alpha != 0 else .99
                alpha = alpha if alpha > .05 else .05
            else:
                alpha = self.alpha

            # Calculate new Q-values
            bellman = ((1 - alpha) * prev_q) + alpha*(reward + self.gamma*(this_q))
            self.q.loc[[self.prev_state], self.prev_action] = bellman

            self.visit.loc[[self.prev_state], self.prev_action] += 1

        # (2) Save and return action/state
        self.prev_state  = state
        self.prev_action = action

    def reset(self):
        self.prev_state = 0

    def save_model(path=None):
        if path != None:
            self.q.to_csv(path + "-q.csv", sep = ";")
            self.visit.to_csv(path + "-visits.csv", sep = ";")
        elif self.model != None:
            self.q.to_csv(self.model + "-q.csv", sep = ";")
            self.visit.to_csv(self.model + "-visits.csv", sep = ";")
