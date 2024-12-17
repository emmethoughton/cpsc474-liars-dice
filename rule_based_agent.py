'''
Ruled-based Agent for Liar's Dice
Author: Nicolas Liu
As of: December 17, 2024

Final Project for CPSC 474 at Yale University
Professor James Glenn

=== Description ===
Below is an implementation of a heuristic agent for the two-player "Liar's Dice" game. The rules are implemented in sections:
first, a screening to determine whether challenging is optimall; second, a randomization of bluffing; finally, deciding
reasonable raises based on the observable dice and previous bids.

=== Example Usage ===
# Position where you have 5 dice showing (1, 3, 3, 5, 5), opponent has 4 dice, and opponent opened bidding with "two 5s"
position_a = liars_dice.initial_info_set(5, 4, (1, 0, 2, 0, 2, 0), [(2, 5)])
# Estimate the best move for player 1 in position_a for 10 seconds
print(rule_based_agent.find_heuristic_move(position_a))

--- Evaluation in this Repository on the Zoo ---
>$ make
>$ pypy3 evaluate_agents.py
'''

from math import comb
import math
import random
from liars_dice import NUM_FACES, PLAYER_ONE_DICE, PLAYER_TWO_DICE

def prob_bid_is_good(info_set, bid, known_dice):
	'''
	Calculates the probability that a given bid is true given observable dice, independent of bid history

    :param bid: the bid to analyze
    :param known_dice: a set of counts representing the observable dice
	'''
	if bid is None:
		return None
	
	prob = 0
	bid_count = bid[0]
	bid_value = bid[1]
	my_dice = known_dice
	num_opp_dice = info_set.player_two_num_dice

	# find the remaining number of unknown dice needed for the bid to be valid
	remaining_dice_needed = bid_count - my_dice[bid_value - 1]
	
	# checks if observable dice already validate the bid
	if remaining_dice_needed <= 0:
		prob = 1
	# checks if it is impossible for the bid count to be matched
	elif remaining_dice_needed > num_opp_dice: 
		prob = 0
	# calculates the probability of the opponent having enough dice to validate the bid
	else:
		for i in range(num_opp_dice, remaining_dice_needed - 1, -1):
			prob += (1/3)**i * (2/3)**(num_opp_dice - i) * comb(num_opp_dice, i)

	return prob

def true_moves(info_set):
	'''
	Find a set of all moves that are guaranteed to be true, given observable dice

    :param info_set: the current information set
	'''
	moves = info_set.__possible_moves__()
	roll = info_set.player_one_roll

	# search for and only keep possible moves that we have enough dice for
	true_moves = []
	for move in moves:
		if (not move is None) and roll[0] + roll[move[1] - 1] >= move[0]: 
			true_moves.append(move)
	return true_moves

def reasonable_moves(info_set, beta):
	'''
	Find a set of all moves that are "reasonable" to be true, given by the "beta" parameter

    :param info_set: the current information set
	:param beta: hyperparameter for largest reasonable increment of count
	'''
	# find the previous bid count
	if len(info_set.bid_history) == 0:
		prev_bid_count = 0
	else:
		prev_bid_count = info_set.bid_history[-1][0]
	
	moves = info_set.__possible_moves__()
	
	# search for and append moves that raise the count from the previous bid by at most beta
	reasonable_moves = []
	for move in moves:
		if move == None:
			reasonable_moves.append(move)
		elif move[0] <= prev_bid_count + beta:
			reasonable_moves.append(move)
	
	return reasonable_moves

def find_largest_bid(bids):
	'''
	Find the largest bid given a set of bids (highest count, tiebreak by value)

    :param bids: the set of bids to search
	'''
	highest_count = 0
	highest_val = 0
	largest_bid = None

	# searches all bids for the highest
	for bid in bids:
		if bid is None:
			break
		count = bid[0]
		val = bid[1]
		if count == highest_count:
			if val > highest_val:
				highest_val = val
				largest_bid = bid
		elif count > highest_count:
			highest_val = val
			highest_count = count
			largest_bid = bid

	return largest_bid
		

def find_smallest_bid(bids):
	'''
	Find the smallest bid given a set of bids (lowest count, tiebreak by value)

    :param bids: the set of bids to search
	'''
	smallest_count = PLAYER_ONE_DICE + PLAYER_TWO_DICE
	smallest_val = NUM_FACES
	smallest_bid = None

	# searches all bids for the smallest
	for bid in bids:
		if bid is None:
			break
		count = bid[0]
		val = bid[1]
		if count == smallest_count:
			if val < smallest_val:
				smallest_val = val
				smallest_bid = bid
		elif count < smallest_count:
			smallest_val = val
			smallest_count = count
			smallest_bid = bid

	return smallest_bid

def one_up(info_set):
	'''
	Return a bid with the highest frequency observable value and smallest count possible for a raise

    :param info_set: the current information set
	'''
	my_roll = info_set.player_one_roll

	# find previous bid information
	if len(info_set.bid_history) == 0:
		prev_bid_count = 0
		prev_bid_val = 0
	else:
		prev_bid = info_set.bid_history[-1]
		prev_bid_count = prev_bid[0]
		prev_bid_val = prev_bid[1]
	
	highest_known_value = 0
	highest_known_count = 0

	# find the highest frequency value in our roll, ignoring 1 (since 1s are wild)
	for value in range(2, len(my_roll) + 1):
		total_EV = my_roll[value - 1]
		if value == prev_bid_val:
			total_EV += math.floor(prev_bid_count / 2) # estimate how many the opponent has
		if total_EV > highest_known_count:
			highest_known_count = total_EV
			highest_known_value = value
	
	# makes the smallest count raise possible
	if highest_known_value > prev_bid_val:
		one_up_bid = (prev_bid_count, highest_known_value)
	else:
		one_up_bid = (prev_bid_count + 1, highest_known_value)

	return one_up_bid

def find_heuristic_move(info_set):
	'''
	Returns the best move to make using certain heuristic rules

    :param info_set: the current information set
	'''
	# define hyperparameters
	alpha = 0.01 # tolerance for an "unreasonable" previous bid
	beta = 2 # largest increment in count "reasonable" for next bid
	epsilon = 0.01 # bluff percentage
	gamma = 1 # raise percentage (optimized at 1 for conservative players)

	# previous bid variables
	if len(info_set.bid_history) == 0:
		prev_bid = None
		prev_bid_count = 0
	else:
		prev_bid = info_set.bid_history[-1]
		prev_bid_count = prev_bid[0]
		prev_bid_val = prev_bid[1]
		prob_last_bid_was_good = prob_bid_is_good(info_set, prev_bid, info_set.player_one_roll)

	# generate tiers of movesets
	moves = info_set.__possible_moves__()
	reasonables = reasonable_moves(info_set, beta)
	trues = true_moves(info_set)
	
	# trivial boundary conditions for previous bid validity
	if not prev_bid is None and prob_last_bid_was_good <= alpha: # if previous bid is trivially or "reasonably" false, then challenge
		return None
	elif not prev_bid is None and prob_last_bid_was_good == 1: # if previous bid is trivially true, challenge is not an option
		if None in reasonables:
			reasonables.remove(None)
		if None in trues:
			trues.remove(None) 
	
	# otherwise, consider reasonable bids
	one_up_bid = one_up(info_set)
	
	# reasonable raises
	if len(reasonables) > 0:
		moves = reasonables
	if len(moves) == 1:
		return moves[0]
	# smallest_reasonable = find_smallest_bid(moves)
	largest_reasonable = find_largest_bid(moves) # largest_reasonable deemed to be optimal
	# random_reasonable = moves[random.randint(0, len(moves) - 1)]
	
	# guaranteed to be true raises
	if len(trues) > 0:
		# smallest_true = find_smallest_bid(trues)
		largest_true = find_largest_bid(trues) # largest_true deemed to be optimal
		# random_true = random.choice(trues)

	r = random.random()
	if r < epsilon:
		return largest_reasonable
	elif len(trues) > 0:
		return largest_true
	else: # will have to start bluffing or making reasonable assumptions - one_up_bid
		return one_up_bid
