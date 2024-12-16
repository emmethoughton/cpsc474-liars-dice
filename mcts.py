'''
MCTS Algorithm for Liar's Dice
Author: Emmet Houghton
As of: December 11, 2024

Final Project for CPSC 474 at Yale University
Professor James Glenn

Below is an implementation of a form of single-observer MCTS for the "Liar's Dice" game. 
The algorithm applies UCB to choose moves for the observer while sampling determinizations with a heuristic strategy
for player two, given information about their rolls is hidden to the observer.

--- Example Usage ---
# Position where you have 5 dice showing (1, 3, 3, 5, 5), opponent has 4 dice, and opponent opened bidding with "two 5s"
position_a = mcts.initial_info_set(5, 4, (1, 0, 2, 0, 2, 0), [(2, 5)])
# Run MCTS on a random starting position where both players have 3 dice
position_b = mcts.initial_info_set(3, 3)
# Estimate the best move for player 1 in position_a for 10 seconds
print(mcts.mcts(position_a, 10))

--- Evaluation in this Repository on the Zoo ---
>$ make
>$ pypy3 evaluate_agents.py
'''

import time
import math
import random
from itertools import product

NUM_FACES = 6
EPSILON = 0.2

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
    
class ISNode:
    def __init__(self, info_set):
        '''
        Builds a node storing total reward, visit count, and edges to children
        :param info_set: the information set represented by the node
        '''
        self.info_set = info_set
        self.edges = None
        self.visit_count = 0
        self.total_reward = 0

    def expand(self, node_memo):
        '''
        Extend the subtree rooted at this node by 1 level
        :param node_memo: a dictionary of existing nodes (for easy state aggregation)
        '''
        self.edges = []
        for action in self.info_set.__possible_moves__():
            succ = self.info_set.__successor__(action)
            if succ in node_memo:
                child = node_memo[succ]
            else:
                child = ISNode(succ)
                node_memo[succ] = child
            self.edges.append(Edge(action, child))

    def is_expanded(self):
        '''
        True if this node has been expanded
        '''
        return not self.info_set.player_one_turn or not self.edges is None

    def is_terminal(self):
        '''
        True if this node is terminal
        '''
        return self.info_set.__is_terminal__()
    
    def ucb_choice(self):
        '''
        Choose the move based on the ucb algorithm (maximize exploit + explore or minimize exploit - explore)
        '''
        if self.is_expanded():
            if (self.visit_count == 0):
                return random.choice(self.edges)
            return self.edges[argmax([e.ucb(self.visit_count, 1) for e in self.edges])]
        else:
            return None

def generate_roll_tuples(num_dice, size=NUM_FACES):
    '''
    Generates all possible rolls
    :param num_dice: the number of dice to roll
    :param size: the number of faces on each die
    '''
    candidates = product(range(num_dice + 1), repeat=size)
    valid_tuples = [t for t in candidates if sum(t) == num_dice]
    return valid_tuples

def bayesian_determinization_distribution(player_two_num_dice, bid_history, root_player_one_turn):
    '''
    Determines the bayesian probabilities of each determinization assuming use of the epsilon-conservative strategy.
    :param player_two_num_dice: the number of dice player two is holding
    :param bid_history: the bidding history of the game
    :param root_player_one_turn: True iff player one bid first in the game
    '''
    all_rolls = generate_roll_tuples(player_two_num_dice)
    weights = [1] * len(all_rolls)
    for i in range(len(all_rolls)):
        current_turn_p1 = root_player_one_turn
        for bid in bid_history:
            if not current_turn_p1:
                num_faces = all_rolls[i][bid[1] - 1] + all_rolls[i][0]
                if num_faces >= bid[0]:
                    weights[i] *= 1/EPSILON
            current_turn_p1 = not current_turn_p1
    total = sum(weights)
    weights = [weights[i] / float(total) for i in range(len(weights))]
    return all_rolls, weights

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

class Edge:
    def __init__(self, action, child):
        '''
        Constructs an edge for a tree node
        :param action: the move which the edge represents
        :param child: a pointer to the successor state (for easy traversal)
        :param n: the number of times this edge has been selected by the UCB algorithm
        '''
        self.action = action
        self.child = child
        self.n = 0
    
    def ucb(self, T, actor):
        '''
        Calculate ucb value of choosing the action along this edge
        :param T: the total number of times the parent node has been visited
        :param actor: 1 for max nodes, -1 for min nodes
        '''
        if self.n == 0 or self.child.visit_count == 0:
            return float('inf') * actor
        return self.child.total_reward / float(self.child.visit_count) + math.sqrt(2*math.log(T)/float(self.n)) * actor

def mcts(info_set, time_limit):
    '''
    The central algorithm. Runs MCTS on a subtree for time_limit and returns the algorithm's choice of move.
    :param info_set: The information set of the player to move at the root of the subtree
    :param time_limit: The amount of time afforded to the MCTS algorithm
    '''
    # No legal moves at termal state
    if info_set.__is_terminal__():
        raise ValueError("MCTS root state must be non-terminal.")
    
    root = ISNode(info_set)
    node_memo = {info_set: root}
    start_time = time.time()
    # Sample determinizations based on the opponent's bidding history
    all_rolls, probabilities = bayesian_determinization_distribution(info_set.player_two_num_dice, info_set.bid_history, len(info_set.bid_history) % 2 == 0)
    while time.time() - start_time < time_limit:
        determinization = random.choices(all_rolls, weights=probabilities, k=1)[0]
        traverse(root, node_memo, determinization)

    if not root.is_expanded():
        return random.choice(root.info_set.__possible_moves__())
    # Choose the move with the highest average reward over the MCTS traversals
    return root.edges[argmax([e.child.total_reward / float(e.child.visit_count) if e.child.visit_count > 0 else float('-inf') for e in root.edges])].action

# DEPRECATED: Previously used for determinization belief sampling, but approach was too slow for large numbers of bids
def traverse_through_node(root, node, node_memo, determinization):
    '''
    Only pursues tree traversals which go through the node parameter.
    Does NOT update MCTS statistics until node is reached.
    :param root: the root of the game tree
    :param node: the node to traverse through
    :param node_memo: a dictionary of nodes (to generalize for state aggregation easily)
    :determinization: the opponent's determinized roll
    '''
    bid_prefix = node.info_set.bid_history
    if len(root.info_set.bid_history) >= len(bid_prefix):
        return traverse(root, node_memo, determinization)
    
    if root.info_set.player_one_turn:
        action = bid_prefix[len(root.info_set.bid_history)]
        next_info_set = root.info_set.__successor__(action)
    else:
        action = epsilon_conservative(determinization, root.info_set.__possible_moves__())
        if action != bid_prefix[len(root.info_set.bid_history)]:
            return
        next_info_set = root.info_set.__successor__(action)
    if not next_info_set in node_memo:
        node_memo[next_info_set] = ISNode(next_info_set)
    traverse_through_node(node_memo[next_info_set], node, node_memo, determinization)

def traverse(root, node_memo, determinization):
    '''
    Traverse the subtree rooted at param root while updating MCTS statistics
    :param root: the root of the subtree
    :param node_memo: a dictionary of nodes (to generalize for state aggregation easily)
    :param determinization: the opponent's roll
    '''
    if root.is_terminal():
        # Base case: propagate the reward back up the tree
        reward = score(root.info_set.player_one_roll, determinization, root.info_set.bid_history, root.info_set.player_one_turn)
        root.total_reward += reward
        root.visit_count += 1
        return reward
    if not root.info_set.player_one_turn:
        # Exploit + explore for player 2 using a heuristic "epsilon-conservative" strategy
        action = epsilon_conservative(determinization, root.info_set.__possible_moves__())
        next_info_set = root.info_set.__successor__(action)
        if not next_info_set in node_memo:
            node_memo[next_info_set] = ISNode(next_info_set)
        reward = traverse(node_memo[next_info_set], node_memo, determinization)
    elif root.is_expanded():
        # Exploit + explore for player 1 using UCB algorithm
        ucb_edge = root.ucb_choice()
        ucb_edge.n += 1
        reward = traverse(ucb_edge.child, node_memo, determinization)
    else:
        # Base case #2: conduct a random playout, then propagate the reward up the tree 
        root.expand(node_memo)
        random_edge = random.choice(root.edges)
        random_edge.n += 1
        random_child = random_edge.child
        reward = random_play(random_child.info_set, determinization)
        random_child.visit_count += 1
        random_child.total_reward += reward
    root.visit_count += 1
    root.total_reward += reward
    return reward

def random_play(info_set, determinization):
    ''' 
    Simulate a random playout from a certain state to a terminal state
    :param info_set: player one's information at the start of the playout
    :param determinization: player two's roll
    '''
    current_is = info_set
    while not current_is.__is_terminal__():
        current_is = current_is.__successor__(random.choice(current_is.__possible_moves__()))
    return score(info_set.player_one_roll, determinization, info_set.bid_history, info_set.player_one_turn)

# Computes the first index with a maximal element in iterable
def argmax(iterable):
    max_index, max_value = 0, iterable[0]
    for index, value in enumerate(iterable):
        if value > max_value:
            max_index, max_value = index, value
    return max_index

# Returns a random index with a maximal element in iterable
def rand_argmax(iterable):
    max_index, max_value = [], iterable[0]
    for index, value in enumerate(iterable):
        if value > max_value:
            max_index, max_value = [index], value
        elif value == max_value:
            max_index.append(index)
    return random.choice(max_index)