#!/bin/bash
clear
# parallel --eta --ungroup python3 run_ppo.py -c cfg/ant/cbp.yml -s {1} ::: $(seq 0 19) &
# parallel --eta --ungroup python3 run_ppo.py -c cfg/ant/l2.yml -s {1} ::: $(seq 0 19) &
# parallel --eta --ungroup python3 run_ppo.py -c cfg/ant/ns.yml -s {1} ::: $(seq 0 19) &
# parallel --eta --ungroup python3 run_ppo.py -c cfg/ant/redo.yml -s {1} ::: $(seq 0 19) &
# parallel --eta --ungroup python3 run_ppo.py -c cfg/ant/std.yml -s {1} ::: $(seq 0 19) &

for i in `seq 0 20`;
do
	j=$((i+0))
	taskset -c $j python3 run_ppo.py -c cfg/ant/cbp.yml -s $i &
done

for i in `seq 0 20`;
do
	j=$((i+20))
	taskset -c $j python3 run_ppo.py -c cfg/ant/l2.yml -s $i &
done

for i in `seq 0 20`;
do
	j=$((i+40))
	taskset -c $j python3 run_ppo.py -c cfg/ant/ns.yml -s $i &
done

for i in `seq 0 20`;
do
	j=$((i+60))
	taskset -c $j python3 run_ppo.py -c cfg/ant/redo.yml -s $i &
done

for i in `seq 0 20`;
do
	j=$((i+80))
	taskset -c $j python3 run_ppo.py -c cfg/ant/std.yml -s $i &
done