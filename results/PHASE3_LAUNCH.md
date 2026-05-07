# Phase 3 Launch Procedure

The phase 3 launch must be detached from the brev SSH session so the loop survives connection cycling.

## Prep
1. Wipe smoke artefacts so production CSV starts clean:
   ```
   brev exec step-5 "docker exec vscode bash -c 'rm -f /workspace/probes/results/probe_results.csv /workspace/probes/results/ood_metrics.csv /workspace/probes/STATUS.md /tmp/probe_phase3.log /tmp/phase3.done'"
   ```
2. Confirm scripts are current on step-5 (mtime later than last edit):
   ```
   brev exec step-5 "docker exec vscode bash -c 'ls -la /workspace/probes/'"
   ```

## Launch (detached)
   ```
   brev exec step-5 "docker exec -d vscode bash -c 'cd /workspace/probes && bash run_phase3.sh tier12 ; echo \$? > /tmp/phase3.done'"
   ```

`docker exec -d` returns immediately; the bash subshell + python.sh process tree is owned by the docker daemon and survives `brev exec` SSH cycling.

## Polling
Every 30 min:
   ```
   brev exec step-5 "docker exec vscode bash -c 'cat /workspace/probes/STATUS.md; echo; tail -5 /tmp/probe_phase3.log'"
   ```

## Done check
When `/tmp/phase3.done` exists, phase 3 is complete:
   ```
   brev exec step-5 "docker exec vscode bash -c 'cat /tmp/phase3.done; wc -l /workspace/probes/results/probe_results.csv'"
   ```

Expected counts (Tier 1+2 with linear at all 3 milestones, MLP + raw_input at 100% only):
- 27 linear_l2 rows
- 9 mlp_l2 rows (one per 100% checkpoint)
- 9 raw_input rows (one per 100% checkpoint)
- Total: 45 rows
