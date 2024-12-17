import random

NUM_FACES = 6
EPSILON = 0.2
PLAYER_ONE_DICE = 3
PLAYER_TWO_DICE = 3

class LiarsDiceIS:
    def __init__(self, player_one_num_dice, player_two_num_dice, player_one_roll, bid_history, player_one_turn):
        '''
        Initializes the state for a two-player Liar's Dice game.
        
        :param player_one_num_dice: The number of dice held by player 1
        :param player_two_num_dice: The number of dice held by player 2
        :param player_one_roll: A tuple of NUM_FACES integers which add to NUM_DICE representing the count
                                of each possible roll in order: (# of 1s (wild), # of 2s, ..., # of 6s)
                                Example: ((1, 2, 0, 1, 1, 0), (0, 1, 1, 0, 0, 3))
        :param bid_history: A list of tuples representing the bids as (quantity, face value). 
                            Example: [(3, 5), None] means the opening bid was "three dice showing 5",
                            and the second player immediately challenged
        :param player_one_turn: An integer which is True if it is player one's turn
        '''
        
        self.player_one_num_dice = player_one_num_dice
        self.player_two_num_dice = player_two_num_dice
        self.player_one_roll = player_one_roll
        self.bid_history = bid_history
        self.player_one_turn = player_one_turn

    def __is_terminal__(self):
        '''
        True if the state is terminal (same for every determinization)
        '''
        return len(self.bid_history) > 0 and self.bid_history[-1] is None
    
    def __is_chance__(self):
        '''
        True if the current state is a chance node
        '''
        return self.player_one_roll is None

    def __possible_moves__(self):
        '''
        Finds all possible bids which could be appended to bid_history from a given state (same for all determinizations)
        '''
        if self.__is_terminal__():
            return []
        
        # Previous move is the starting point. If no bids yet, the lowest possible bid is a single 2.
        possible_moves = []
        if len(self.bid_history) == 0:
            possible_moves.append((1, 2))
            last_bid = (1, 2)
        else:
            last_bid = self.bid_history[-1]
        # Bids of the same quantity but a higher face value are allowed
        for face_value in range(last_bid[1] + 1, NUM_FACES + 1):
            possible_moves.append((last_bid[0], face_value))
        # Any bid of a higher quantity is allowed
        for quantity in range(last_bid[0] + 1, self.player_one_num_dice + self.player_two_num_dice + 1):
            for face_value in range(2, NUM_FACES + 1):
                possible_moves.append((quantity, face_value))
        # If there has been a bid, you can challenge it
        if len(self.bid_history) > 0:
            possible_moves.append(None)
            
        return possible_moves

    def __successor__(self, bid):
        '''
        State transition function
        :param bid: The bid made by the current player
        '''
        new_history = self.bid_history.copy()
        new_history.append(bid)
        return LiarsDiceIS(self.player_one_num_dice, self.player_two_num_dice, self.player_one_roll, new_history, not self.player_one_turn)

    def __str__(self):
        '''
        Returns a string representation of the current state for debugging.     
        '''
        return (f"Player One Rolls: {self.player_one_roll}\n"
                f"Bid History: {self.bid_history}\n"
                f"Player One's Turn? {self.player_one_turn}")

    def __eq__(self, other):
        '''
        Override: deep comparison for dictionary collisions
        :param other: the object to compare to
        '''
        if isinstance(other, LiarsDiceIS):
            return (self.bid_history == other.bid_history)
        return False

    def __hash__(self):
        '''
        Override: hashes bid history as info sets (in a given call to MCTS) are uniquely determined by bids 
        '''
        return hash(tuple(self.bid_history))

def initial_info_set(player_one_num_dice, player_two_num_dice, player_one_roll=None, bid_history=[], player_one_turn=True):
    '''
    Generates the initial state with NUM_DICE random rolls for player one
    :param player_one_num_dice: the number of dice player one is holding
    :param player_two_num_dice: the number of dice player two is holding
    :param player_one_roll: the number of each face player one is looking at (random by default)
    :param bid_history: the bids which have already been made (empty by default)
    :param player_one_turn: DEPRECATED (True if player one is the mover)
    '''
    if player_one_roll is None:
        roll = [random.randint(1, NUM_FACES) for _ in range(player_one_num_dice)]
        player_one_roll = tuple(roll.count(face) for face in range(1, NUM_FACES + 1))
    # Player one is the player to move at the root of the MCTS tree
    return LiarsDiceIS(player_one_num_dice, player_two_num_dice, player_one_roll, bid_history, player_one_turn)

def epsilon_conservative(roll, moves, epsilon=EPSILON):
    '''
    Heuristic strategy for choosing player two's actions during the tree traversal.
    With probability 1-EPSILON, chooses a random raise compatible with visible dice or challenges if not possible.
    With probability EPSILON, chooses an arbitrary "bluff" (helps the algorithm explore)
    :param info_set: the info set of the current node
    :param determinization: player two's roll
    :param epsilon: a hyperparameter adjusting the opponent's probability of bluffing
    '''
    r = random.random()
    if r < epsilon:
        if len(moves) == 1:
            return moves[0]
        return moves[random.randint(0, len(moves) - 1)]
    else:
        viable_moves = []
        for move in moves:
            if (not move is None) and roll[0] + roll[move[1] - 1] >= move[0]:
                viable_moves.append(move)
        if len(viable_moves) > 0:
            return random.choice(viable_moves)
        else:
            return None

def score(player_one_roll, player_two_roll, bid_history, player_one_last_bidder):
    '''
    Scores a (terminal) position for player one.
    The game is zero sum, so +1 means player one gains a dice, and -1 means player one loses a dice.
    :param player_one_roll: player one's dice
    :param player_two_roll: player two's dice
    :param bid_history: the bidding history of the game
    :param player_one_last_bidder: True if player two challenged, False otherwise
    '''

    # Bidding history must have ended in a challenge
    if len(bid_history) == 0 or not bid_history[-1] is None:
        return 0
    
    last_bid = bid_history[-2]
    # Scoring based on the total number of dice of the face value of the last bid + wilds
    num_faces = player_one_roll[last_bid[1] - 1] + player_one_roll[0] + player_two_roll[last_bid[1] - 1] + player_two_roll[0]
    if player_one_last_bidder:
        return 1 if num_faces >= last_bid[0] else -1
    else:
        return -1 if num_faces >= last_bid[0] else 1
