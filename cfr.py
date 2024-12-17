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
        self.info_key = info_key
        self.moves = moves
        self.total_regret = {move: 0 for move in moves}
        self.strategy = {move: 0 for move in moves}
        self.total_prob = {move: 0 for move in moves}
    
    def get_curr_strat(self, probability):
        total_pos_regret = 0
        for move in self.moves:
            self.strategy[move] = max(0, self.total_regret[move])
            total_pos_regret += self.strategy[move]
        if total_pos_regret > 0:
            for move in self.moves:
                self.strategy[move] /= total_pos_regret
                self.total_prob[move] += probability * self.strategy[move]
        else:
            for move in self.moves:
                self.strategy[move] = 1 / len(self.moves)
                self.total_prob[move] += probability * self.strategy[move]
        return self.strategy
    
    def get_average_strat(self):
        average_strat = {move: 0 for move in self.moves}
        total_prob = 0
        for move in self.moves:
            total_prob += self.total_prob[move]
        if total_prob > 0:
            for move in self.moves:
                average_strat[move] = self.total_prob[move] / total_prob
        else:
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
    # Generate all possible rolls for n dice
    all_rolls = product(range(1, 7), repeat=n)
    
    # Count frequencies of dice values for each roll
    outcomes = Counter(tuple(roll.count(i) for i in range(1, 7)) for roll in all_rolls)
    
    # Calculate probabilities for each outcome
    probabilities = {
        outcome: (factorial(n) / (factorial(outcome[0]) *
                                  factorial(outcome[1]) *
                                  factorial(outcome[2]) *
                                  factorial(outcome[3]) *
                                  factorial(outcome[4]) *
                                  factorial(outcome[5])) * (1/6)**n)
        for outcome in outcomes
    }
    
    return probabilities

def get_cfr(info_set, time_limit):
    start_time = time.time()
    PLAYER_ONE_DICE = info_set.player_one_num_dice
    PLAYER_TWO_DICE = info_set.player_two_num_dice

    iterations = 0
    info_map = {}
    while time.time() - start_time < time_limit:
        iterations += 1
        # initialize other player with no roll
        my_player_info = copy.deepcopy(info_set)
        other_player_info = LiarsDiceIS(PLAYER_TWO_DICE, PLAYER_ONE_DICE, None, my_player_info.bid_history, not my_player_info.player_one_turn)
        # make current_player (0 index) whosever turn it is
        info_sets = [my_player_info, other_player_info]
        # run cfr on game tree
        cfr(info_map, info_sets, [1,1,1])
        print("tree")

    info_key = str(info_set.player_one_roll) + " " + str(info_set.bid_history[-3:])
    my_node = info_map[info_key]
    calc_strategy = my_node.get_average_strat()
    print(f"iterations: {iterations}")
    for move in my_node.moves:
        print(f"move: {move}")
        print(f"prob: {calc_strategy[move]}")

    moves = list(calc_strategy.keys())
    probs = list(calc_strategy.values())
    # return move based on average strategy from all iterations
    return random.choices(moves, weights=probs, k=1)[0]

def cfr(info_map, info_sets, probs):
    if info_sets[0].__is_terminal__() or info_sets[1].__is_terminal__():
        if info_sets[0].player_one_turn:
            curr_player = 0
        else:
            curr_player = 1
        score = liars_dice.score(info_sets[curr_player].player_one_roll, info_sets[1 - curr_player].player_one_roll, 
                                info_sets[curr_player].bid_history, info_sets[curr_player].player_one_turn)
        # print(info_sets[curr_player].bid_history)
        # print(f"current player: {curr_player}")
        # print(f"score: {score}")
        return score
    if info_sets[0].__is_chance__() or info_sets[1].__is_chance__():
        expected_outcome = 0
        if info_sets[0].__is_chance__():
            other_player = 0
        else:
            other_player = 1
        
        # calculate all possible dice rolls for other player
        dice_probs_other = generate_dice_outcomes(info_sets[other_player].player_one_num_dice)
        for outcome_other, prob_other in dice_probs_other.items():
            info_sets[other_player] = copy.deepcopy(info_sets[other_player])
            info_sets[other_player].player_one_roll = outcome_other
            expected_outcome += prob_other * cfr(info_map, info_sets, 
                [probs[0], probs[1], probs[2] * prob_other])
        
        # expected_outcome = cfr(info_map, info_sets, probs)
        return expected_outcome
    
    if info_sets[0].player_one_turn:
        curr_player = 0
    else:
        curr_player = 1

    info_key = str(info_sets[curr_player].player_one_roll) + " " + str(info_sets[curr_player].bid_history[-3:])
    moves = info_sets[curr_player].__possible_moves__()

    if info_key not in info_map:
        info_map[info_key] = CFRNode(info_key, moves)
    curr_strat = info_map[info_key].get_curr_strat(probs[curr_player])
    # print(curr_strat)
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

def cfr_policy(time_limit):
    def func(info_set):
        return get_cfr(info_set, time_limit)
    return func

if __name__ == "__main__":
    sample_info_set = initial_info_set(3, 3, (0, 0, 0, 0, 3, 0), [(5,1)], True)
    get_cfr(sample_info_set, 2)
