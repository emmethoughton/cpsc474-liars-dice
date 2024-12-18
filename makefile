TestLiarsDice:
	echo "#!/bin/bash" > LiarsDice
	echo "python3 test_agents.py \"\$$@\"" >> LiarsDice
	chmod u+x LiarsDice
