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
from liars_dice import epsilon_conservative, score, NUM_FACES, EPSILON
    
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
