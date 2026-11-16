# Evaluation scope

[`robotwin-50.csv`](../results/robotwin-50.csv) contains 100 task-condition results: 50 tasks, each evaluated for 100 episodes in `demo_clean` and 100 in `demo_randomized`. These are the released Action-only results. Success counts are integers; percentages and condition means are derived from them.

[`summary.json`](../results/summary.json) records the aggregate values and the CSV digest. The comparison column contains the authors' reported fine-tuned scores and links to their source. It is not a same-stack rerun of the fine-tuned checkpoint.

[`base-control-19.json`](../results/base-control-19.json) contains a separate frozen-Base control on a fixed 19-task subset. Report differences only over the available matched task-condition cells. Do not extrapolate that control to all 50 benchmark tasks, or interpret the results as proving that a changed starting point accounts for every difference from fine-tuning.

The release does not present partial Joint evaluations as a complete benchmark, and does not mix diagnostic reruns into the registered Action-only scores.
