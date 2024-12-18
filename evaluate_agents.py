'''
Heuristics, MCTS, and CFR for Liar's Dice
Authors: Emmet Houghton, Nicolas Liu, Tyler Tan
As of: December 17, 2024

Final Project for CPSC 474 at Yale University
Professor James Glenn

=== Brief Description ===
Liar's Dice is an imperfect information, turn-based game in which a group of players take turns bidding "increasing" lower bounds 
for the total number of a particular face value across the dice they roll collectively. No player can see their opponents' dice. 
If one player deems an opponent's bid sufficiently unreasonable, they may call "Liar!" and the bid is settled. The loser of the 
challenge loses a dice before going into the next round. For this project, we focus on the heads up version of Liar's dice in which
ones are wild.

For this project, we develop a CFR agent, an SO-MCTS agent, and a rule-based agent for a heads up version of Liar's Dice. 
Counterfactual regret (CFR) minimization is the most rigorous algorithm for determining the optimal policy for imperfect information 
games such as two-player Liar's Dice, but it requires significant convergence time. In developing these algorithms, we are particularly
curious about the relative performance of these agents under time pressure.

For evaluating our agents, we also built two baseline agents: a very naive random agent and an "epsilon-conservative" heuristic agent,
which chooses a bluff with probability epsilon and chooses an arbitrary raise compatible with its own dice with probability (1-epsilon).
(If there is no such bid, the agent challenges.) This agent was also used in traversing the MCTS tree because it balances exploitation
and exploration reasonably and provides solid approximations for the probabilities of each determinization when playing against a human
player. After training, we also play the rule-based, MCTS, and CFR agents against each other to measure relative performance, taking
into account training times and other parameters.

=== Research Question ===
How do CFR and MCTS agents perform against a rule-based agent in Liar's Dice? Under what time constraints do these agents exhibit 
strategic strength?

=== Summary of Results ===
All 3 agents perform similarly against the random agent (>90% win rate over 100 games) and 
the epsilon-conservative agent (>75% win rate over 100 games).

Against each other, the MCTS agent does notably better against the rule-based agent (65% win rate over 100 games).

The results take roughly 20 minutes to obtain for the MCTS and CFR agents.

=== How to run this testing script: ===
>$ make
>$ ./LiarsDice
or simply
>$ pypy3 evaluate_agents.py
'''

import liars_dice
from liars_dice import NUM_FACES, PLAYER_ONE_DICE, PLAYER_TWO_DICE
import mcts
import rule_based_agent
import random

def simulate_game(policy_a, policy_b):
    rolls_one = [random.randint(1, NUM_FACES) for _ in range(PLAYER_ONE_DICE)]
    rolls_two = [random.randint(1, NUM_FACES) for _ in range(PLAYER_TWO_DICE)]

    counts_one = tuple(rolls_one.count(face) for face in range(1, NUM_FACES + 1))
    counts_two = tuple(rolls_two.count(face) for face in range(1, NUM_FACES + 1))

    #print("Player 1 Rolls:", counts_one)
    #print("Player 2 Rolls:", counts_two)

    current_position_a = liars_dice.initial_info_set(PLAYER_ONE_DICE, PLAYER_TWO_DICE, counts_one, bid_history=[], player_one_turn=True)
    current_position_b = liars_dice.initial_info_set(PLAYER_TWO_DICE, PLAYER_ONE_DICE, counts_two, bid_history=[], player_one_turn=False)
    mover_a = True
    while not current_position_a.__is_terminal__():
        if mover_a:
            next_action = policy_a(current_position_a)
        else:
            next_action = policy_b(current_position_b)
        current_position_a = current_position_a.__successor__(next_action)
        current_position_b = current_position_b.__successor__(next_action)
        mover_a = not mover_a
    #print("Bid History:", current_position_a.bid_history)
    score = liars_dice.score(counts_one, counts_two, current_position_a.bid_history, len(current_position_a.bid_history) % 2 == 0)
    #print("Result:", score)
    return score

# define policies to test
mcts_policy_2sec = lambda info_set: mcts.mcts(info_set, 2)
random_policy = lambda info_set: random.choice(info_set.__possible_moves__())
epsilon_conservative_heuristic = lambda info_set: liars_dice.epsilon_conservative(info_set.player_one_roll, info_set.__possible_moves__())
rule_based = lambda info_set: rule_based_agent.find_heuristic_move(info_set)

# run simulated games
NUM_SIMULATIONS = 10
total_score_random = 0
for i in range(NUM_SIMULATIONS):
    total_score_random += max(simulate_game(rule_based, random_policy), 0)
total_score_epscons = 0
for i in range(NUM_SIMULATIONS):
    total_score_epscons += max(simulate_game(rule_based, epsilon_conservative_heuristic), 0)

print("===== EVALUATION RESULTS =====")
print("MCTS Result against Random Play:", total_score_random / NUM_SIMULATIONS)
print("MCTS Result against Epsilon-Conservative:", total_score_epscons / NUM_SIMULATIONS)
