'''
Monte-Carlo CFR Agent for Liar's Dice
Author: Tyler Tan
As of: December 17, 2024

Final Project for CPSC 474 at Yale University
Professor James Glenn

=== Description ===
Below is an implementation of a Monte-Carlo CFR agent for the two-player "Liar's Dice" game. 

=== Example Usage ===
# Position where you have 5 dice showing (1, 3, 3, 5, 5), opponent has 4 dice, and opponent opened bidding with "two 5s"
position_a = liars_dice.initial_info_set(5, 4, (1, 0, 2, 0, 2, 0), [(2, 5)])
# Estimate the best move for player 1 in position_a for 10 seconds
print(monte_cfr.get_monte_cfr(position_a, 10))
'''
from liars_dice import LiarsDiceIS, initial_info_set
import liars_dice
import time
import copy
from cfr import CFRNode, generate_dice_outcomes, get_move_from_strat

def get_monte_cfr(info_set, time_limit):
    '''
        Returns an action/next move from a given information set
        based on the Monte-Carlo Counterfactual Regret Algorithm
        :param info_set: the information set of a player
        :param time_limit: the number of seconds to run CFR 
    '''

    start_time = time.time()
    PLAYER_ONE_DICE = info_set.player_one_num_dice
    PLAYER_TWO_DICE = info_set.player_two_num_dice

    iterations = 0
    info_map = {}
    while time.time() - start_time < time_limit:
        my_player_info = copy.deepcopy(info_set)

        # initialize other player with no roll
        other_player_info = LiarsDiceIS(PLAYER_TWO_DICE, PLAYER_ONE_DICE, None, my_player_info.bid_history, not my_player_info.player_one_turn)
        info_sets = [my_player_info, other_player_info]

        # run monte_cfr on game tree
        monte_cfr(info_map, info_sets, [1,1,1])

        iterations += 1

    info_key = str(info_set.player_one_roll) + " " + str(info_set.bid_history[-1:])
    my_node = info_map[info_key]
    calc_strategy = my_node.get_average_strat()

    # print(f"iterations: {iterations}")
    # for move in my_node.moves:
    #     print(f"move: {move}")
    #     print(f"prob: {calc_strategy[move]}")

    # return move based on average strategy from all iterations
    return get_move_from_strat(calc_strategy)

def monte_cfr(info_map, info_sets, probs):
    '''
        Returns the expected utility given the current state
        in the game tree. Randomly chooses opponents action and dice roll.
        :param info_map: a dictionary of information sets to CFRNodes
        :param info_sets: an array of information sets (0 for player, 1 for opponent)
        :param probs: an array of probabilities [p0, p1, chance]
    '''

    # terminal state
    if info_sets[0].__is_terminal__() or info_sets[1].__is_terminal__():
        if info_sets[0].player_one_turn:
            curr_player = 0
        else:
            curr_player = 1
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
        
        # sample random dice roll for other player based on probability of roll
        dice_probs_other = generate_dice_outcomes(info_sets[other_player].player_one_num_dice)
        outcome_other = get_move_from_strat(dice_probs_other)
        info_sets[other_player].player_one_roll = outcome_other
        
        action_prob = dice_probs_other[outcome_other]
        expected_outcome += action_prob * monte_cfr(info_map, info_sets, 
                [probs[0], probs[1], probs[2] * action_prob])
        return expected_outcome
    
    if info_sets[0].player_one_turn:
        curr_player = 0
    else:
        curr_player = 1

    info_key = str(info_sets[curr_player].player_one_roll) + " " + str(info_sets[curr_player].bid_history[-1:])
    moves = info_sets[curr_player].__possible_moves__(True)

    if info_key not in info_map:
        info_map[info_key] = CFRNode(info_key, moves)
    curr_strat = info_map[info_key].get_curr_strat(probs[curr_player])

    # player node
    if curr_player == 0:
        payoff = 0
        pay = {move:0 for move in moves}
        for move in moves:
            # make move and update bid history/player turn
            succ_0 = info_sets[0].__successor__(move)
            succ_1 = info_sets[1].__successor__(move)

            # calculate expected payoff from move
            pay[move] = -1 * monte_cfr(info_map, [succ_0, succ_1], 
                    [probs[0] * curr_strat[move], probs[1], probs[2]])

            # add to total expected payoff
            payoff += curr_strat[move] * pay[move]

        # sums up total regret
        other_player = 1 - curr_player
        for move in moves:
            info_map[info_key].total_regret[move] +=  probs[other_player] * probs[2] * (pay[move] - payoff)
        
        return payoff
    # opponent node
    else:
        # get random move based off current strategy and play it
        move = get_move_from_strat(curr_strat)
        succ_0 = info_sets[0].__successor__(move)
        succ_1 = info_sets[1].__successor__(move)
        return -1 * monte_cfr(info_map, [succ_0, succ_1], 
                    [probs[0], probs[1] * curr_strat[move], probs[2]])

if __name__ == "__main__":
    sample_info_set = initial_info_set(2, 2, (0, 0, 0, 1, 1, 0), [], True)
    get_monte_cfr(sample_info_set, 10)
