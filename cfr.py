'''
CFR Agent for Liar's Dice
Author: Tyler Tan
As of: December 17, 2024

Final Project for CPSC 474 at Yale University
Professor James Glenn

=== Description ===
Below is an implementation of a CFR agent for the two-player "Liar's Dice" game. 

=== Example Usage ===
# Position where you have 5 dice showing (1, 3, 3, 5, 5), opponent has 4 dice, and opponent opened bidding with "two 5s"
position_a = liars_dice.initial_info_set(5, 4, (1, 0, 2, 0, 2, 0), [(2, 5)])
# Estimate the best move for player 1 in position_a for 10 seconds
print(cfr.get_cfr(position_a))
'''
from liars_dice import LiarsDiceIS, initial_info_set
import liars_dice
import time
import random
from itertools import product
from math import factorial
from collections import Counter
import copy

class CFRNode:
    def __init__(self, info_key, moves):
        '''
        Builds a node storing info_map key, possible moves, strategy, and total regret
        for current player
        :param info_key: the key for this node in info_map
        :param moves: all possible moves from last bid
        '''
        self.info_key = info_key
        self.moves = moves
        self.total_regret = {move: 0 for move in moves}
        self.strategy = {move: 0 for move in moves}
        self.total_prob = {move: 0 for move in moves}
    
    def get_curr_strat(self, probability):
        '''
        Calculate current strategy based off total regret
        :param probability: the probability that the current player
        got to this node
        '''
        total_pos_regret = 0
        # get toal positive regret
        for move in self.moves:
            self.strategy[move] = max(0, self.total_regret[move])
            total_pos_regret += self.strategy[move]
        
        # average over cumulative pos regret
        if total_pos_regret > 0:
            for move in self.moves:
                self.strategy[move] /= total_pos_regret
                self.total_prob[move] += probability * self.strategy[move]
        # uniform distribution
        else:
            for move in self.moves:
                self.strategy[move] = 1 / len(self.moves)
                self.total_prob[move] += probability * self.strategy[move]
        return self.strategy
    
    def get_average_strat(self):
        '''
        Calculate average strategy based off total probability
        '''
        average_strat = {move: 0 for move in self.moves}
        total_prob = 0
        for move in self.moves:
            total_prob += self.total_prob[move]
        if total_prob > 0:
            for move in self.moves:
                average_strat[move] = self.total_prob[move] / total_prob
        else:
            # uniform
            for move in self.moves:
                average_strat[move] = 1 / len(self.moves)
        # renormalize
        normal_prob = 0
        for move in self.moves:
            if average_strat[move] < 0.001:
                average_strat[move] = 0
            else:
                normal_prob += average_strat[move]
        for move in self.moves:
            average_strat[move] / normal_prob
        return average_strat
    
def generate_dice_outcomes(n):
    '''
        Generate the probabilities for all dice rolls
        :param n: number of dice
    '''
    # Generate all possible rolls for n dice
    all_rolls = product(range(1, 7), repeat=n)
    
    # count for dice roll
    outcomes = Counter(tuple(roll.count(i) for i in range(1, 7)) for roll in all_rolls)
    
    # Calculate probabilities for each outcome
    probabilities = {}
    for outcome in outcomes:
        probabilities[outcome] = (factorial(n) / (factorial(outcome[0]) *
                                        factorial(outcome[1]) *
                                        factorial(outcome[2]) *
                                        factorial(outcome[3]) *
                                        factorial(outcome[4]) *
                                        factorial(outcome[5])) * (1/6)**n)
    
    return probabilities

def get_move_from_strat(strat):
    '''
        Returns a bid/challenge based on given probabilities of a strategy
        :param strat: a dictionary that maps an action to a probability
    '''
    moves = list(strat.keys())
    probs = list(strat.values())
    return random.choices(moves, weights=probs, k=1)[0]

def get_cfr(info_set, time_limit):
    '''
        Returns an action/next move from a given information set
        based on the Counterfactual Regret Algorithm
        :param info_set: the information set of a player
        :param time_limit: the number of seconds to run CFR 
    '''
    start_time = time.time()
    PLAYER_ONE_DICE = info_set.player_one_num_dice
    PLAYER_TWO_DICE = info_set.player_two_num_dice

    iterations = 0
    info_map = {}
    while time.time() - start_time < time_limit:
        # initialize other player with no roll
        my_player_info = copy.deepcopy(info_set)
        other_player_info = LiarsDiceIS(PLAYER_TWO_DICE, PLAYER_ONE_DICE, None, my_player_info.bid_history, not my_player_info.player_one_turn)

        # array of information sets
        info_sets = [my_player_info, other_player_info]

        # run cfr on game tree
        cfr(info_map, info_sets, [1,1,1])

        # count iterations
        iterations += 1

    # key corresponding to given information set
    info_key = str(info_set.player_one_roll) + " " + str(info_set.bid_history[-1:])
    # get strategy calculated for that information set
    my_node = info_map[info_key]
    calc_strategy = my_node.get_average_strat()
    
    # # print probability distribution
    # print(f"iterations: {iterations}")
    # for move in my_node.moves:
    #     print(f"move: {move}")
    #     print(f"prob: {calc_strategy[move]}")

    return get_move_from_strat(calc_strategy)

def cfr(info_map, info_sets, probs):
    '''
        Returns the expected utility given the current state
        in the game tree
        :param info_map: a dictionary of information sets to CFRNodes
        :param info_sets: an array of information sets (0 for player, 1 for opponent)
        :param probs: an array of probabilities [p0, p1, chance]
    '''

    # check if at a terminal state
    if info_sets[0].__is_terminal__() or info_sets[1].__is_terminal__():
        if info_sets[0].player_one_turn:
            curr_player = 0
        else:
            curr_player = 1
        
        # calculate score based off current player
        score = liars_dice.score(info_sets[curr_player].player_one_roll, info_sets[1 - curr_player].player_one_roll, 
                                info_sets[curr_player].bid_history, info_sets[curr_player].player_one_turn)
        return score
    
    # chance node
    if info_sets[0].__is_chance__() or info_sets[1].__is_chance__():
        expected_outcome = 0
        if info_sets[0].__is_chance__():
            other_player = 0
        else:
            other_player = 1
        
        # calculate all possible dice rolls for other player
        dice_probs_other = generate_dice_outcomes(info_sets[other_player].player_one_num_dice)
        for outcome_other, prob_other in dice_probs_other.items():
            # assign dice roll
            info_sets[other_player] = copy.deepcopy(info_sets[other_player])
            info_sets[other_player].player_one_roll = outcome_other
            expected_outcome += prob_other * cfr(info_map, info_sets, 
                [probs[0], probs[1], probs[2] * prob_other])
        
        return expected_outcome
    
    if info_sets[0].player_one_turn:
        curr_player = 0
    else:
        curr_player = 1

    # get key for info_map
    info_key = str(info_sets[curr_player].player_one_roll) + " " + str(info_sets[curr_player].bid_history[-1:])
    moves = info_sets[curr_player].__possible_moves__(True)

    # create CFRNode if not in map
    if info_key not in info_map:
        info_map[info_key] = CFRNode(info_key, moves)
    curr_strat = info_map[info_key].get_curr_strat(probs[curr_player])
    
    payoff = 0
    pay = {move:0 for move in moves}
    for move in moves:
        # make move and update bid history/player turn
        succ_0 = info_sets[0].__successor__(move)
        succ_1 = info_sets[1].__successor__(move)

        # calculate expected payoff from move
        if curr_player == 0:
            pay[move] = -1 * cfr(info_map, [succ_0, succ_1], 
                [probs[0] * curr_strat[move], probs[1], probs[2]])
        else:
            pay[move] = -1 * cfr(info_map, [succ_0, succ_1], 
                [probs[0], probs[1] * curr_strat[move], probs[2]])

        # add to total expected payoff
        payoff += curr_strat[move] * pay[move]

    # sums up total regret
    other_player = 1 - curr_player
    for move in moves:
        info_map[info_key].total_regret[move] +=  probs[other_player] * probs[2] * (pay[move] - payoff)
    
    return payoff


if __name__ == "__main__":
    sample_info_set = initial_info_set(3, 3, (0, 0, 0, 0, 3, 0), [(5,1)], True)
    get_cfr(sample_info_set, 2)
