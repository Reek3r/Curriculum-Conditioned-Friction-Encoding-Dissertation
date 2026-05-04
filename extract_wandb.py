import wandb

api = wandb.Api()
run = api.run("m-roomanwaheed-the-university-of-manchester/WheeledLab/k1hj5qh1")

print(f"Run: {run.name} | State: {run.state}\n")

full = run.history(keys=[
    "Train/mean_reward",
    "Episode_Reward/b_lap",
    "Episode/track_progress_completed",
    "Episode_Termination/out_of_bounds",
    "Episode/episode_length_s",
    "Episode/adr_mu_lo",
    "Episode/adr_mu_hi",
    "_step"
], samples=10000)

checkpoints = [500, 1000, 1500]

for cp in checkpoints:
    nearest = full.iloc[(full["_step"] - cp).abs().argsort()[:1]]
    row = nearest.iloc[0]
    step = int(row["_step"])

    mu_lo = row.get("Episode/adr_mu_lo")
    mu_hi = row.get("Episode/adr_mu_hi")

    print(f"iter ~{step}")
    print(f"  mean_reward                        : {row.get('Train/mean_reward', float('nan')):.2f}")
    print(f"  b_lap                              : {row.get('Episode_Reward/b_lap', float('nan')):.2f}")
    print(f"  track_progress_completed           : {row.get('Episode/track_progress_completed', float('nan')):.2f}")
    print(f"  out_of_bounds                      : {row.get('Episode_Termination/out_of_bounds', float('nan')):.2f}")
    print(f"  episode_length_s                   : {row.get('Episode/episode_length_s', float('nan')):.2f}")
    if mu_lo is not None and mu_hi is not None:
        gap = mu_hi - mu_lo
        fires = max(0, int(round((gap - 0.08) / 0.04)))
        print(f"  adr_mu_lo / adr_mu_hi              : {mu_lo:.2f} / {mu_hi:.2f}  (gap={gap:.3f}, fires={fires})")
    else:
        print(f"  adr_mu_lo / adr_mu_hi              : N/A")
    print()

