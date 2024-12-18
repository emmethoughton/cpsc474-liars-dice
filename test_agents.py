
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
All 3 agents are extremely strong against the random agent (>90% win rate over 100 games) and the epsilon-conservative agent (>80% win
rate over 100 games). 

The MCTS agent finds strong bids for games with at most 10 dice (standard heads up games) in less time than any human's reaction time, 
meaning that time pressure does not pose a significant challenge for this algorithm in a live game scenario. In fact, the performance
of the 0.1 second and 1 second MCTS agents are extremely similar (win rate of 1 second agent is not significantly greater than 50%).

The MCTS agent outperforms the Rule-Based Agent (~65% win rate over 100 games) with alternating first mover and 5-dice each. Even from 
a one die disadvantage, the MCTS agent remains competitive, but the rule-based agent has a strong record with the advantage of the first
move in an equal-dice game.

Qualitatively, we note that the MCTS agent plays more aggressively than the rule-based agent, particularly on early bids, suggesting
human players underestimate the value of making bids with less confidence. Such a phenomenon in observing players new to poker.

=== How to run this testing script (quick results, ~2 minutes): ===
>$ make
>$ ./LiarsDice
or simply
>$ pypy3 test_agents.py

=== How to run this testing script (complete results, ~45 minutes): ===
1. Uncomment #complete_results() in the main function at the bottom of this file
2. Comment out quick_results() in the main function to avoid duplicate results
3. Run testing script as for the quick results
'''

import liars_dice
from liars_dice import NUM_FACES
import mcts
import rule_based_agent
import cfr
import random

def simulate_game(policy_a, policy_b, a_dice, b_dice):
    rolls_one = [random.randint(1, NUM_FACES) for _ in range(a_dice)]
    rolls_two = [random.randint(1, NUM_FACES) for _ in range(b_dice)]

    counts_one = tuple(rolls_one.count(face) for face in range(1, NUM_FACES + 1))
    counts_two = tuple(rolls_two.count(face) for face in range(1, NUM_FACES + 1))

    current_position_a = liars_dice.initial_info_set(a_dice, b_dice, counts_one, bid_history=[], player_one_turn=True)
    current_position_b = liars_dice.initial_info_set(b_dice, a_dice, counts_two, bid_history=[], player_one_turn=False)
    mover_a = True
    while not current_position_a.__is_terminal__():
        if mover_a:
            next_action = policy_a(current_position_a)
        else:
            next_action = policy_b(current_position_b)
        current_position_a = current_position_a.__successor__(next_action)
        current_position_b = current_position_b.__successor__(next_action)
        mover_a = not mover_a
    score = liars_dice.score(counts_one, counts_two, current_position_a.bid_history, len(current_position_a.bid_history) % 2 == 0)
    return score

def matchup(policy_a, policy_b, a_dice, b_dice, num_simulations, label, alternate=True):
    policy_a_total_score = 0
    for i in range(num_simulations):
        if alternate and i % 2 == 1:
            policy_a_total_score += 1 - max(simulate_game(policy_b, policy_a, a_dice, b_dice), 0)
        else:
            policy_a_total_score += max(simulate_game(policy_a, policy_b, a_dice, b_dice), 0)
    print(label, policy_a_total_score/num_simulations)

def quick_results():
	'''
     A set of quick (~2 minutes) evaluations done on our developed agents.
     '''
	# define policies to test
	mcts_policy_tenthsec = lambda info_set: mcts.mcts(info_set, 0.1)
	mcts_policy_onesec = lambda info_set: mcts.mcts(info_set, 1)
	rule_based = rule_based_agent.find_heuristic_move
	cfr_policy = cfr.cfr_policy(1)
	random_policy = lambda info_set: random.choice(info_set.__possible_moves__())
	epsilon_conservative_heuristic = lambda info_set: liars_dice.epsilon_conservative(info_set.player_one_roll, info_set.__possible_moves__())

	print("===== EVALUATION RESULTS =====")
	print("--- Example Moves from Each Agent: ---")
	print("Suppose you have 5 dice showing (1, 3, 3, 5, 5), opponent has 4 dice, and opponent opened bidding with 'two 5s':")
	position_a = liars_dice.initial_info_set(5, 4, (1, 0, 2, 0, 2, 0), [(2, 5)])
	print(position_a, "\n")
	print("MCTS(1 sec) move choice:", mcts_policy_onesec(position_a))
	print("Rule-Based Agent move choice:", rule_based(position_a))
	#print("CFR(1 sec) move choice:", cfr_policy(position_a))

	print("Suppose you have 3 dice showing (1, 2, 6), your opponent has 5 dice, and you have the first move:")
	position_b = liars_dice.initial_info_set(3, 5, (1, 1, 0, 0, 0, 1), [])
	print(position_b, "\n")
	print("MCTS(1 sec) move choice:", mcts_policy_onesec(position_b))
	print("Rule-Based Agent move choice:", rule_based(position_b))
	#print("CFR(1 sec) move choice:", cfr_policy(position_a))

	print("\n--- Heads Up Win Rates (10-game matchups for illustration): ---")
	NUM_SIMULATIONS = 10
	matchup(mcts_policy_tenthsec, random_policy, 5, 5, NUM_SIMULATIONS, "MCTS(0.1 sec) v. Random, alternating first mover, 5 dice each:")
	matchup(mcts_policy_tenthsec, epsilon_conservative_heuristic, 5, 5, NUM_SIMULATIONS, "MCTS(0.1 sec) v. Epsilon-Conservative, alternating first mover, 5 dice each:")
	matchup(mcts_policy_tenthsec, mcts_policy_onesec, 5, 5, NUM_SIMULATIONS, "MCTS(0.1 sec) v. MCTS(1 sec), alternating first mover, 5 dice each:")
	matchup(rule_based, random_policy, 5, 5, NUM_SIMULATIONS, "Rule-Based v. Random, alternating first mover, 5 dice each:")
	matchup(rule_based, epsilon_conservative_heuristic, 5, 5, NUM_SIMULATIONS, "Rule-Based v. Epsilon-Conservative, alternating first mover, 5 dice each:")
	matchup(mcts_policy_tenthsec, rule_based, 5, 5, NUM_SIMULATIONS, "MCTS(0.1 sec) v. Rule-Based, alternating first mover, 5 dice each:")
	matchup(mcts_policy_tenthsec, rule_based, 5, 5, NUM_SIMULATIONS, "Rule-Based v. MCTS(0.1sec), Rule-Based is first mover, 5 dice each:", alternate=False)
	matchup(mcts_policy_onesec, rule_based, 3, 4, NUM_SIMULATIONS, "MCTS(1 sec) v. Rule-Based, alternating first mover, 3 and 4 dice respectively:")
	matchup(mcts_policy_onesec, rule_based, 3, 2, NUM_SIMULATIONS, "MCTS(1 sec) v. Rule-Based, alternating first mover, 3 and 2 dice respectively:")
	matchup(cfr_policy, random_policy, 1, 1, NUM_SIMULATIONS, "CFR(1 sec) v. Random, alternating first mover, 1 dice each:")
	matchup(cfr_policy, epsilon_conservative_heuristic, 1, 1, NUM_SIMULATIONS, "CFR(1 sec) v. Epsilon-Conservative, alternating first mover, 1 dice each:")

def complete_results():
	'''
     A set of complete (~45 minutes) evaluations done on our developed agents. Notable extensions from quick_results() are:
     increased number of simulated games from 10 to 100 (to decrease variance), increased CFR time limit from 1 to 5 seconds
     '''
	# define policies to test
	mcts_policy_tenthsec = lambda info_set: mcts.mcts(info_set, 0.1)
	mcts_policy_onesec = lambda info_set: mcts.mcts(info_set, 1)
	rule_based = rule_based_agent.find_heuristic_move
	cfr_policy = cfr.cfr_policy(5)
	random_policy = lambda info_set: random.choice(info_set.__possible_moves__())
	epsilon_conservative_heuristic = lambda info_set: liars_dice.epsilon_conservative(info_set.player_one_roll, info_set.__possible_moves__())

	print("===== EVALUATION RESULTS =====")
	print("\n--- Heads Up Win Rates (100-game matchups for illustration): ---")
	NUM_SIMULATIONS = 100
	matchup(mcts_policy_tenthsec, random_policy, 5, 5, NUM_SIMULATIONS, "MCTS(0.1 sec) v. Random, alternating first mover, 5 dice each:")
	matchup(mcts_policy_tenthsec, epsilon_conservative_heuristic, 5, 5, NUM_SIMULATIONS, "MCTS(0.1 sec) v. Epsilon-Conservative, alternating first mover, 5 dice each:")
	matchup(mcts_policy_tenthsec, mcts_policy_onesec, 5, 5, NUM_SIMULATIONS, "MCTS(0.1 sec) v. MCTS(1 sec), alternating first mover, 5 dice each:")
	matchup(rule_based, random_policy, 5, 5, NUM_SIMULATIONS, "Rule-Based v. Random, alternating first mover, 5 dice each:")
	matchup(rule_based, epsilon_conservative_heuristic, 5, 5, NUM_SIMULATIONS, "Rule-Based v. Epsilon-Conservative, alternating first mover, 5 dice each:")
	matchup(mcts_policy_tenthsec, rule_based, 5, 5, NUM_SIMULATIONS, "MCTS(0.1 sec) v. Rule-Based, alternating first mover, 5 dice each:")
	matchup(mcts_policy_tenthsec, rule_based, 5, 5, NUM_SIMULATIONS, "Rule-Based v. MCTS(0.1sec), Rule-Based is first mover, 5 dice each:", alternate=False)
	matchup(mcts_policy_onesec, rule_based, 3, 4, NUM_SIMULATIONS, "MCTS(1 sec) v. Rule-Based, alternating first mover, 3 and 4 dice respectively:")
	matchup(mcts_policy_onesec, rule_based, 3, 2, NUM_SIMULATIONS, "MCTS(1 sec) v. Rule-Based, alternating first mover, 3 and 2 dice respectively:")
	matchup(cfr_policy, random_policy, 1, 1, NUM_SIMULATIONS, "CFR(5 sec) v. Random, alternating first mover, 2 dice each:")
	matchup(cfr_policy, epsilon_conservative_heuristic, 1, 1, NUM_SIMULATIONS, "CFR(5 sec) v. Epsilon-Conservative, alternating first mover, 2 dice each:")
	matchup(cfr_policy, mcts_policy_tenthsec, 1, 1, NUM_SIMULATIONS, "CFR(5 sec) v. MCTS(0.1 sec)), alternating first mover, 2 dice each:")

if __name__ == "__main__":
     quick_results()
	 #complete_results()
