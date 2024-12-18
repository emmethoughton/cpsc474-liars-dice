
'''
Heuristics, MCTS, and CFR for Liar's Dice
Authors: Emmet Houghton, Nicolas Liu, Tyler Tan
As of: December 17, 2024

Final Project for CPSC 474 at Yale University
Professor James Glenn

=== Brief Description of Game ===
Liar's Dice is an imperfect information, turn-based game in which a group of players take turns bidding "increasing" lower bounds 
for the total number of a particular face value across the dice they roll collectively. No player can see their opponents' dice. 
If one player deems an opponent's bid sufficiently unreasonable, they may call "Liar!" and the bid is settled. The loser of the 
challenge loses a dice before going into the next round. For this project, we focus on the heads up version of Liar's dice in which
ones are wild.

For this project, we develop a CFR agent, an SO-MCTS agent, and a rule-based agent for a heads up version of Liar's Dice. 
Counterfactual regret (CFR) minimization is the most rigorous algorithm for determining the optimal policy for imperfect information 
games such as two-player Liar's Dice, but it requires significant convergence time. In developing these algorithms, we are particularly
curious about the relative performance of these agents under time pressure. (In fact, to reduce convergence time, we also end up 
developing a Monte Carlo CFR Agent--see results below.)

For evaluating our agents, we use two baseline agents: a very naive random agent and an "epsilon-conservative" heuristic agent,
which chooses a bluff with probability epsilon and chooses an arbitrary raise compatible with its own dice with probability (1-epsilon).
(If there is no such bid, the agent challenges.) This agent was also used in traversing the MCTS tree because it balances exploitation
and exploration reasonably and provides solid approximations for the probability distribution of each determinization when playing 
against a human player. After training, we also play the rule-based, MCTS, and CFR agents against each other to measure relative 
performance, taking into account training times and other parameters.

=== Research Question ===
How do CFR and MCTS agents perform against a rule-based agent in Liar's Dice? Under what time constraints do these agents exhibit 
strategic strength?

=== Summary of Results ===
Results regarding the tradeoff between convergence time and strategic strength:

- The MCTS agent finds strong bids for games with 10 dice (standard heads up games) in less than any human's reaction time, meaning 
that time pressure does not pose a significant challenge for this algorithm in a live game scenario. In fact, the performance
of the 0.1 second and 1 second MCTS agents are extremely similar (win rate of 1 second agent is not significantly greater than 50%
over 100 games). This result suggests that UCB typically determines the best move in a 10-dice game with confidence within a tenth
of a second.

- The CFR agent, on the other hand, takes significant time to converge for games with higher quantities of dice; one complete traversal
of the game tree takes on the order of 30 minutes for 3 dice per player. We include results for this agent in games where each player 
has 2 dice below, but this finding led us to develop a stochastic version of the CFR agent which chooses moves following the Monte-Carlo
Counterfactual Regret Algorithm to traverse the tree. This algorithm exhibits strength in games with 3 dice each.

Results regarding relative performance of agents:

Both the MCTS and Rule-Based agents are extremely strong against the random agent (>95% win rate over 100 games) and the epsilon-conservative 
agent (>85% win rate over 100 games). Precisely, we have (win rates are reported for the first algorithm in the matchup):
MCTS(0.1 sec) v. Random, alternating first mover, 5 dice each: 0.98
MCTS(0.1 sec) v. Epsilon-Conservative, alternating first mover, 5 dice each: 0.92
Rule-Based v. Random, alternating first mover, 5 dice each: 0.98
Rule-Based v. Epsilon-Conservative, alternating first mover, 5 dice each: 0.91

The MCTS agent outperforms the Rule-Based Agent (~60% win rate over 100 games) with alternating first mover and 5-dice each. Even from 
a one die disadvantage, the MCTS agent remains competitive, but the rule-based agent has a strong record with the advantage of the first
move in an equal-dice game:
MCTS(0.1 sec) v. Rule-Based, alternating first mover, 5 dice each: 0.59
MCTS(1 sec) v. Rule-Based, alternating first mover, 3 and 4 dice respectively: 0.72
Rule-Based v. MCTS(0.1sec), *Rule-Based is always first mover*, 5 dice each: 0.6

The CFR agent does quite well against the random agent (~85% win rate over 100 games) but only slightly outperforms the epsilon-conservative 
and rule-based agents (~50-60% win rate over 100 games). This is likely due to convergence time as 5 seconds is likely not enough time to 
get a large amount of iterations of the entire game tree, especially if the player has to make the opening bid. Even when restricting 
possible moves to increasing the quantity by at most 1, the runtime for 1 iteration still took more than 10 minutes.
CFR(5 sec) v. Random, alternating first mover, 2 dice each: 0.85
CFR(5 sec) v. Epsilon-Conservative, alternating first mover, 2 dice each: 0.61
CFR(5 sec) v. Rule-based, alternating first mover, 2 dice each: 0.56

The Monte-Carlo CFR agent tries to remedy the slow convergence time by introducing randomness when selecting the opponent's action
based off the current strategy. This allows for more iterations of the algorithm within the same time frame and prioritizes paths in the 
tree that are more likely to occur. The results show that for each of the other agents (random, epsilon-conservative, rule-based),
there was about a 10% increase in win rate. In the future, to improve these results, instead of a time limit, the agent can specify
the number of iterations to run on the game tree (ex. 10000) to ensure convergence for each information set. This means determining moves
for opening bids will be quite slow but as the bid history progresses, it will become exponentially faster.
MONTE_CFR(5 sec) v. Random, alternating first mover, 3 dice each: 0.96
MONTE_CFR(5 sec) v. Epsilon-Conservative, alternating first mover, 3 dice each: 0.76
MONTE_CFR(5 sec) v. Rule-based, alternating first mover, 3 dice each: 0.69

Qualitatively, we note that the MCTS agent plays more aggressively than the rule-based agent, particularly on early bids, suggesting
human players underestimate the value of making bids with less confidence. Such a phenomenon is common in observing players new to poker. 
On the other hand, the CFR and MONTE-CFR agent like to take more defensive approaches since they are converging to Nash Equilibrium,
which is achieved by minimizing regret.

=== How to run this testing script (quick results, ~2 minutes): ===
>$ make
>$ ./LiarsDice
or simply
>$ pypy3 test_agents.py

=== How to run this testing script (complete results, ~5 hours): ===
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
import monte_cfr

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
     A set of quick (~3 minutes) evaluations done on our developed agents.
     '''
	# define policies to test
	mcts_policy_tenthsec = lambda info_set: mcts.mcts(info_set, 0.1)
	mcts_policy_onesec = lambda info_set: mcts.mcts(info_set, 1)
	rule_based = rule_based_agent.find_heuristic_move
	cfr_policy = lambda info_set: cfr.get_cfr(info_set, 1)
	random_policy = lambda info_set: random.choice(info_set.__possible_moves__())
	epsilon_conservative_heuristic = lambda info_set: liars_dice.epsilon_conservative(info_set.player_one_roll, info_set.__possible_moves__())
	monte_cfr_policy = lambda info_set: monte_cfr.get_monte_cfr(info_set, 1)

	print("===== EVALUATION RESULTS =====")
	print("--- Example Moves from Agents: ---")
	print("Suppose you have 5 dice showing (1, 3, 3, 5, 5), opponent has 4 dice, and opponent opened bidding with 'two 5s':")
	position_a = liars_dice.initial_info_set(5, 4, (1, 0, 2, 0, 2, 0), [(2, 5)])
	print(position_a, "\n")
	print("MCTS(1 sec) move choice:", mcts_policy_onesec(position_a))
	print("Rule-Based Agent move choice:", rule_based(position_a))

	print("\nSuppose you have 3 dice showing (1, 2, 6), your opponent has 5 dice, and you have the first move:")
	position_b = liars_dice.initial_info_set(3, 5, (1, 1, 0, 0, 0, 1), [])
	print(position_b, "\n")
	print("MCTS(1 sec) move choice:", mcts_policy_onesec(position_b))
	print("Rule-Based Agent move choice:", rule_based(position_b))

	print("\nSuppose you have 2 dice showing (3, 4), your opponent has 3 dice, and opponent opened bidding with 'one 6':")
	position_c = liars_dice.initial_info_set(2, 3, (0, 0, 1, 1, 0, 0), [(1, 6)])
	print(position_c, "\n")
	print("MCTS(1 sec) move choice:", mcts_policy_onesec(position_c))
	print("Rule-Based Agent move choice:", rule_based(position_c))
	print("MONTE_CFR(1 sec) move choice:", monte_cfr_policy(position_c))

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
	matchup(cfr_policy, epsilon_conservative_heuristic, 1, 1, NUM_SIMULATIONS, "CFR(1 sec) v. Epsilon-Conservative, alternating first mover, 1 die each:")
	matchup(monte_cfr_policy, random_policy, 1, 1, NUM_SIMULATIONS, "MONTE_CFR(1 sec) v. Random, alternating first mover, 1 die each:")
	matchup(monte_cfr_policy, epsilon_conservative_heuristic, 1, 1, NUM_SIMULATIONS, "MONTE_CFR(1 sec) v. Epsilon-Conservative, alternating first mover, 1 die each:")
	matchup(monte_cfr_policy, mcts_policy_tenthsec, 1, 1, NUM_SIMULATIONS, "MONTE_CFR(1 sec) v. MCTS(0.1sec), alternating first mover, 1 die each:")

def complete_results():
	'''
     A set of complete (~45 minutes) evaluations done on our developed agents. Notable extensions from quick_results() are:
     increased number of simulated games from 10 to 100 (to decrease variance), increased CFR time limit from 1 to 10 seconds
     '''
	# define policies to test
	mcts_policy_tenthsec = lambda info_set: mcts.mcts(info_set, 0.1)
	mcts_policy_onesec = lambda info_set: mcts.mcts(info_set, 1)
	rule_based = rule_based_agent.find_heuristic_move
	cfr_policy = lambda info_set: cfr.get_cfr(info_set, 5)
	monte_cfr_policy = lambda info_set: monte_cfr.get_monte_cfr(info_set, 5)
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
	matchup(cfr_policy, random_policy, 2, 2, NUM_SIMULATIONS, "CFR(5 sec) v. Random, alternating first mover, 2 dice each:")
	matchup(cfr_policy, epsilon_conservative_heuristic, 2, 2, NUM_SIMULATIONS, "CFR(5 sec) v. Epsilon-Conservative, alternating first mover, 2 dice each:")
	matchup(cfr_policy, rule_based, 2, 2, NUM_SIMULATIONS, "CFR(5 sec) v. Rule-based, alternating first mover, 2 dice each:")
	matchup(monte_cfr_policy, random_policy, 3, 3, NUM_SIMULATIONS, "MONTE_CFR(5 sec) v. Random, alternating first mover, 3 dice each:")
	matchup(monte_cfr_policy, epsilon_conservative_heuristic, 3, 3, NUM_SIMULATIONS, "MONTE_CFR(5 sec) v. Epsilon-Conservative, alternating first mover, 3 dice each:")
	matchup(monte_cfr_policy, rule_based, 3, 3, NUM_SIMULATIONS, "MONTE_CFR(5 sec) v. Rule-based, alternating first mover, 3 dice each:")

if __name__ == "__main__":
     quick_results()
	#  complete_results()
