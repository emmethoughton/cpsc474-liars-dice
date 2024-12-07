import random

NUM_FACES = 6

def initial_state(player_one_num_dice, player_two_num_dice):
    '''
    Generates the initial state with NUM_DICE random rolls per player
    '''
    rolls_one = [random.randint(1, NUM_FACES) for _ in range(player_one_num_dice)]
    rolls_two = [random.randint(1, NUM_FACES) for _ in range(player_two_num_dice)]
    counts_one = tuple(rolls_one.count(face) for face in range(1, NUM_FACES + 1))
    counts_two = tuple(rolls_two.count(face) for face in range(1, NUM_FACES + 1))
    return LiarsDiceState(counts_one, counts_two, [], True)

class LiarsDiceState:
    def __init__(self, player_one_num_dice, player_two_num_dice, player_one_roll, player_two_roll, bid_history, player_one_turn):
        '''
        Initializes the state for a two-player Liar's Dice game.
        
        :param player_one_roll: A tuple of NUM_FACES integers which add to NUM_DICE representing the count
                                of each possible roll in order: (# of 1s (wild), # of 2s, ..., # of 6s)
                                Example: ((1, 2, 0, 1, 1, 0), (0, 1, 1, 0, 0, 3))
        :param player_two_roll: The same for the second player
        :param current_bid: A list of tuples representing the bids as (quantity, face value). 
                            Example: [(3, 5), None] means the opening bid was "three dice showing 5",
                            and the second player immediately challenged
        :param current_turn: An integer which is True if it is player one's turn
        '''
        
        self.num_dice = (player_one_num_dice, player_two_num_dice)
        self.player_one_roll = player_one_roll
        self.player_two_roll = player_two_roll
        self.bid_history = bid_history
        self.player_one_turn = player_one_turn

    def __is_terminal__(self):
        return len(self.bid_history) > 0 and self.bid_history[-1] is None

    def __score__(self):
        '''
        Scores the (terminal) state for playerone . The game is zero sum, so +1 means player 1 gains
        a dice, and -1 means player 1 loses a dice.
        '''
        if not self.__is_terminal__():
            return 0
        
        last_bid = self.bid_history[-2]
        # Scoring based on the total number of dice of the face value of the last bid + wilds
        num_faces = self.player_one_roll[last_bid[1] - 1] + self.player_one_roll[0] + self.player_two_roll[last_bid[1] - 1] + self.player_two_roll[0]
        if self.player_one_turn:
            return 1 if num_faces >= last_bid[0] else -1
        else:
            return -1 if num_faces >= last_bid[0] else 1

    def __possible_moves__(self):
        '''
        Finds all possible bids which could be appended to bid_history from a given state
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
        for quantity in range(last_bid[0] + 1, sum(self.num_dice) * 2 + 1):
            for face_value in range(1, NUM_FACES + 1):
                possible_moves.append((quantity, face_value))
        # If there has been a bid, you can challenge it
        if len(self.bid_history) > 0:
            possible_moves.append(None)
            
        return possible_moves

    def __move__(self, bid):
        '''
        State transition function
        :param bid: The bid made by the current player
        '''
        new_history = self.bid_history.copy()
        new_history.append(bid)
        return LiarsDiceState(self.player_one_roll, self.player_two_roll, new_history, not self.player_one_turn)

    def __str__(self):
        '''
        Returns a string representation of the current state for debugging.     
        '''
        return (f"Player Rolls: {self.player_one_roll}\n"
                f"Player Rolls: {self.player_two_roll}\n"
                f"Bid History: {self.bid_history}\n"
                f"Player One's Turn? {self.player_one_turn}")

