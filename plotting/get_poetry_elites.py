import json
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

PLOT_OUTPUT = "qd_score.png"

def plot_std_error_mean(data, run_name, ax):
    # pad if necessary
    pad = len(max(data, key=len))
    data = np.array([i + [i[-1]]*(pad-len(i)) for i in data]) # pad with last value of each seed data
    # data.shape => [num_seeds, timesteps]
    mean = np.mean(data, axis=0)
    std_error = np.std(data, axis=0) / np.sqrt(data.shape[0])
    print(mean[-1])
    print(std_error[-1])

    # Create a list of timesteps (or x-axis values) based on the length of one of the lists
    timesteps = list(range(data.shape[1]))

    ax.plot(timesteps, mean, label=run_name)

    # Use fill_between() to show standard error lines, add 1.96 z factor for 95% CI
    ax.fill_between(timesteps, mean - (std_error*1.96), mean + (std_error*1.96), alpha=0.2)

def load_run_stats(
    base_dirs: list[str],
) -> dict[str, Any]:
    qd_score_runs = {}
    for base_dir in base_dirs: # contains every run in base_dir
        qd_score_seeds = []
        for seed_dir in base_dir.rglob("**/history.jsonl"):
            with open(seed_dir, "r") as f:
                data = [json.loads(line) for line in f]
                data = data[:500] # get only first 500 iters
            # empty elites archive
            all_genres_tones = [
                f'{g}_{t}' for g in ['haiku', 'sonnet', 'ballad', 'limerick', 'hymn']
                for t in ['happy', 'dark', 'mysterious', 'romantic', 'reflective']
            ]
            elites = {key: ('', -1) for key in all_genres_tones}  # (genotype, fitness) tuple

            load_iterations(qd_score_seeds, data, elites, seed_dir)

            # log final elites
            output_path = seed_dir.parent / "elites_500.jsonl"
            with open(output_path, "w") as file:
                for k, v in elites.items():
                    elite_entry = {
                        'poem': v[0],
                        'quality': v[1],
                        'genre': k.split('_')[0],
                        'tone': k.split('_')[1]
                    }
                    file.write(json.dumps(elite_entry) + "\n")

        qd_score_runs[str(base_dir)] = qd_score_seeds
    return qd_score_runs

def load_iterations(qd_score_seeds, data, elites, seed_dir):
    qd_score_seed = []

    # fill in elites archive
    for datapoint in data:
        poem = datapoint['poem']
        quality = float(datapoint['quality'])
        genre = datapoint['genre']
        tone = datapoint['tone']

        if genre not in ['haiku', 'sonnet', 'ballad', 'limerick', 'hymn']:
            print("erroneous genre detected, skipping")
            continue

        if tone not in ['happy', 'dark', 'mysterious', 'romantic', 'reflective']:
            print("erroneous tone detected, skipping")
            continue

        prev_elite, prev_quality = elites[f'{genre}_{tone}']
        if quality > prev_quality:
            elites[f'{genre}_{tone}'] = (poem, quality)
        # qd score per iteration
        qd_score_seed.append(sum([max(v[1], 0) for v in elites.values()]))
        # qd_score_seed.append(sum([int(v[1]>-1) for v in elites.values()])) # coverage
    print(f"Path: {seed_dir} ========= QD: {qd_score_seed[-1]}")
    qd_score_seeds.append(qd_score_seed)

runs_whitelist = {
    "openelm/logs/poetry/qdaif": "qdaif",
    "openelm/logs/poetry/targeted": "targeted",
    "openelm/logs/poetry/random": "random",
}

qd_score_runs = load_run_stats([Path(list(runs_whitelist.keys())[i]) for i in range(len(runs_whitelist))])

# compute stats for each run/experiment, and add to same plot
fig, ax = plt.subplots()

for run in qd_score_runs.keys():
    if run in runs_whitelist.keys():
        run_data = qd_score_runs[run]
        run_name = runs_whitelist[run]
        plot_std_error_mean(run_data, run_name, ax)

ax.set_xlabel('Iterations')
ax.set_ylabel('QD Score')
# ax.set_ylabel('Coverage')
# ax.set_title(PLOT_TITLE)
ax.legend(loc = "lower right")

plt.savefig(PLOT_OUTPUT)
plt.close('all')
