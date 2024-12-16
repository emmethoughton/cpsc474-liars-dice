import mcts
import time
import random
from itertools import product
from math import factorial
from collections import Counter

# change hashing of info_sets

class StrategyProfile:
    def __init__(self):
        self.total_regret = {}
        self.total_prob = {}
    
    def add_if_absent(self, info_key, moves):
        if info_key not in self.total_regret:
            self.total_regret[info_key] = {move: 0 for move in moves}
            self.total_prob[info_key] = {move: 0 for move in moves}
    
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

    # initialize total regret and probability table for each player
    strats = [StrategyProfile(), StrategyProfile()]
    
    # get current player strat
    if info_set.player_one_turn:
        curr_player_strat = strats[0]
    else:
        curr_player_strat = strats[1]

    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        # initialize other player with no roll
        other_player_info = mcts.LiarsDiceIS(PLAYER_TWO_DICE, PLAYER_ONE_DICE, None, [], not info_set.player_one_turn)
        # make current_player (0 index) whosever turn it is
        if info_set.player_one_turn:
            info_sets = [info_set, other_player_info]
        else:
            info_sets = [other_player_info, info_set]
        # run cfr on game tree
        cfr(info_sets, strats, [1,1,1], 0)
        # print("tree")

    info_key = str(info_set.player_one_roll) + " " + str(info_set.bid_history)
    moves = list(curr_player_strat.total_prob[info_key].keys())
    probs = list(curr_player_strat.total_prob[info_key].values())
    print(f"iterations: {iterations}")
    # for i in range(moves):
    #     print(f"move: {moves[i]}")
    #     print(f"prob: {probs[i]}")

    # return move based on average strategy from all iterations
    return random.choices(moves, weights=[(prob / iterations) for prob in probs], k=1)[0]

def get_curr_strat(strat, moves, key):
    new_strat = {}
    num_moves = len(moves)
    if key not in strat.total_regret:
        for move in moves:
            new_strat[move] = 1 / num_moves
    else:
        total_pos_regret = 0
        regret = strat.total_regret[key]
        # print("yes")
        for move in moves:
            total_pos_regret += max(0, regret[move])
        if total_pos_regret > 0:
            for move in moves:
                new_strat[move] = max(0, regret[move] / total_pos_regret)
        else:
            for move in moves:
                new_strat[move] = 1 / num_moves
    return new_strat

def cfr(info_sets, strats, probs, curr_player):
    if info_sets[curr_player].__is_terminal__():
        score = mcts.score(info_sets[curr_player].player_one_roll, info_sets[1 - curr_player].player_one_roll, 
                                info_sets[curr_player].bid_history, info_sets[curr_player].player_one_turn)
        # print(info_sets[curr_player].bid_history)
        print(f"current player: {curr_player}")
        print(f"score: {score}")
        return score
    if info_sets[curr_player].__is_chance__() or info_sets[1 - curr_player].__is_chance__():
        expected_outcome = 0
        if info_sets[curr_player].__is_chance__():
            other_player = curr_player
        else:
            other_player = 1 - curr_player
        
        # calculate all possible dice rolls for other player
        dice_probs_other = generate_dice_outcomes(info_sets[other_player].player_one_num_dice)
        for outcome_other, prob_other in dice_probs_other.items():
            info_sets[other_player].player_one_roll = outcome_other
            expected_outcome += prob_other * cfr(info_sets, strats, 
                [probs[0], probs[1], probs[2] * prob_other], curr_player)
        # info_sets[other_player].player_one_roll = (3, 0, 0, 0, 0, 0)
        # expected_outcome = cfr(info_sets, strats, probs, curr_player)
        return expected_outcome
    if curr_player == 0:
        pay = {}
        info_key = str(info_sets[0].player_one_roll) + " " + str(info_sets[0].bid_history)
        moves = info_sets[0].__possible_moves__()
        curr_strat = get_curr_strat(strats[0], moves, info_key)
        payoff = 0
        for move in moves:
            # make move and update bid history/player turn
            succ_0 = info_sets[0].__successor__(move)
            succ_1 = info_sets[1].__successor__(move)

            # calculate expected payoff from move
            pay[move] = -1 * cfr([succ_0, succ_1], 
                    strats, [probs[0] * curr_strat[move], probs[1], probs[2]], 1 - curr_player)
            print(f"move: {move} payoff: {pay[move]} curr_strat: {curr_strat[move]}")
            # add to total expected payoff
            payoff += curr_strat[move] * pay[move]

        # print(str(info_sets[0]))
        strats[0].add_if_absent(info_key, moves)
        # sums up total regret and total probability for each move
        for move in moves:
            strats[0].total_regret[info_key][move] +=  probs[1] * probs[2] * (pay[move] - payoff)
            strats[0].total_prob[info_key][move] += probs[0] * curr_strat[move]
        # print(f"payoff: {payoff}")
        return payoff
    else:
        # same for other player
        pay = {}
        moves = info_sets[1].__possible_moves__()
        info_key = str(info_sets[1].player_one_roll) + " " + str(info_sets[1].bid_history)
        curr_strat = get_curr_strat(strats[1], moves, info_key)
        payoff = 0
        for move in moves:
            succ_0 = info_sets[0].__successor__(move)
            succ_1 = info_sets[1].__successor__(move)

            pay[move] = -1 * cfr([succ_0, succ_1], 
                    strats, [probs[0], probs[1] * curr_strat[move], probs[2]], 1 - curr_player)

            payoff += curr_strat[move] * pay[move]

        # print(str(info_sets[1]))
        strats[1].add_if_absent(info_key, moves)
        for move in moves:
            strats[1].total_regret[info_key][move] += probs[0] * probs[2] * (pay[move] - payoff)
            strats[1].total_prob[info_key][move] += probs[1] * curr_strat[move]

        # print(f"payoff: {payoff}")
        return payoff

def cfr_policy(time_limit):
    def func(info_set):
        return get_cfr(info_set, time_limit)
    return func

if __name__ == "__main__":
    sample_info_set = mcts.initial_info_set(3, 3, (3, 0, 0, 0, 0, 0), [], True)
    get_cfr(sample_info_set, 10)