import mcts
import random
from mcts import NUM_FACES

PLAYER_ONE_DICE = 3
PLAYER_TWO_DICE = 3

def simulate_game(policy_a, policy_b):
    rolls_one = [random.randint(1, NUM_FACES) for _ in range(PLAYER_ONE_DICE)]
    rolls_two = [random.randint(1, NUM_FACES) for _ in range(PLAYER_TWO_DICE)]

    counts_one = tuple(rolls_one.count(face) for face in range(1, NUM_FACES + 1))
    counts_two = tuple(rolls_two.count(face) for face in range(1, NUM_FACES + 1))

    print("Player 1 Rolls:", counts_one)
    print("Player 2 Rolls:", counts_two)

    current_position_a = mcts.initial_info_set(PLAYER_ONE_DICE, PLAYER_TWO_DICE, counts_one, bid_history=[])
    current_position_b = mcts.initial_info_set(PLAYER_ONE_DICE, PLAYER_TWO_DICE, counts_two, bid_history=[])
    mover_a = True
    while not current_position_a.__is_terminal__():
        if mover_a:
            next_action = policy_a(current_position_a)
        else:
            next_action = policy_b(current_position_b)
        current_position_a = current_position_a.__successor__(next_action)
        current_position_b = current_position_b.__successor__(next_action)
        mover_a = not mover_a
    print("Bid History:", current_position_a.bid_history)
    score = mcts.score(counts_one, counts_two, current_position_a.bid_history, len(current_position_a.bid_history) % 2 == 0)
    print("Result:", score)
    return score

mcts_policy_2sec = lambda info_set: mcts.mcts(info_set, 2)
random_policy = lambda info_set: random.choice(info_set.__possible_moves__())
epsilon_conservative_heuristic = lambda info_set: mcts.epsilon_conservative(info_set.player_one_roll, info_set.__possible_moves__())

NUM_SIMULATIONS = 10
total_score_random = 0
for i in range(NUM_SIMULATIONS):
    total_score_random += max(simulate_game(mcts_policy_2sec, random_policy), 0)
total_score_epscons = 0
for i in range(NUM_SIMULATIONS):
    total_score_epscons += max(simulate_game(mcts_policy_2sec, epsilon_conservative_heuristic), 0)

print("===== EVALUATION RESULTS =====")
print("MCTS Result against Random Play:", total_score_random / NUM_SIMULATIONS)
print("MCTS Result against Epsilon-Conservative:", total_score_epscons / NUM_SIMULATIONS)
