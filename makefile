TestLiarsDice:
	echo "#!/bin/bash" > TestLiarsDice
	echo "python3 evaluate_agents.py \"\$$@\"" >> TestLiarsDice
	chmod u+x TestLiarsDice
