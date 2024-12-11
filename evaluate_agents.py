import mcts

# Position where you have 5 dice showing (1, 3, 3, 5, 5), opponent has 4 dice, and opponent opened bidding with "two 5s"
position_a = mcts.initial_info_set(5, 4, (1, 0, 2, 0, 2, 0), [(2, 5)])
# Run MCTS on a random starting position where both players have 3 dice
position_b = mcts.initial_info_set(3, 3)
# Find the best move for player 1 in position_a
print(position_a)
print("MCTS Move:", mcts.mcts(position_a, 10))
