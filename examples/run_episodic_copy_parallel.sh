varArgs=$1
CUDA_VISIBLE_DEVICES=0 th episodic_copy_parallel.lua -numNodes 8 -nodeIdx 1 $varArgs &
CUDA_VISIBLE_DEVICES=1 th episodic_copy_parallel.lua -numNodes 8 -nodeIdx 2 $varArgs &
CUDA_VISIBLE_DEVICES=2 th episodic_copy_parallel.lua -numNodes 8 -nodeIdx 3 $varArgs &
CUDA_VISIBLE_DEVICES=3 th episodic_copy_parallel.lua -numNodes 8 -nodeIdx 4 $varArgs &
CUDA_VISIBLE_DEVICES=4 th episodic_copy_parallel.lua -numNodes 8 -nodeIdx 5 $varArgs &
CUDA_VISIBLE_DEVICES=5 th episodic_copy_parallel.lua -numNodes 8 -nodeIdx 6 $varArgs &
CUDA_VISIBLE_DEVICES=6 th episodic_copy_parallel.lua -numNodes 8 -nodeIdx 7 $varArgs &
CUDA_VISIBLE_DEVICES=7 th episodic_copy_parallel.lua -numNodes 8 -nodeIdx 8 $varArgs &
wait