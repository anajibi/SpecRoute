# Operational scripts for the CelebA-HQ ladder

These ran every CelebA-HQ training stage. They exist because three separate failures cost real
time, and each script encodes the fix for one of them.

| script | job |
|---|---|
| `supervise.sh` | runs one training stage; survives session teardown, auto-continues from the newest checkpoint, refuses to restart on a numerical failure |
| `sequence_graphB.sh` | chains stages back to back so the GPU never idles between them |
| `watch.sh` | watches one stage; silent unless something is actionable |
| `watch_seqB.sh` | watches a whole sequence, reporting stage transitions |
| `verify_s3.sh` | proves an uploaded checkpoint is complete, not merely the right size |

## Usage

    # one stage
    systemd-run --user --scope --unit=hdae-<run> --slice=hdae.slice \
      bash ops/supervise.sh <config.yaml> <run-name> <target-step> <s3-prefix>

    # a whole ladder, no idle gaps
    systemd-run --user --scope --unit=hdae-seq --slice=hdae.slice bash ops/sequence_graphB.sh

    # integrity of an uploaded checkpoint
    bash ops/verify_s3.sh najibi-research-7f2a able/celebahq/assets/<run>/<file>.ckpt

`loginctl enable-linger` must be on, or the systemd scope dies with the login session.

## The three failures these encode

**A session teardown killed a run.** `setsid nohup` guards against SIGHUP but not against cgroup
teardown. A run died at step 36,990 after 4.6 h and the GPU sat idle 27 h. Stages now run in their
own systemd scope under `hdae.slice`, outside the session's cgroup.

**`pgrep -f` matches the agent's own shell.** A supervisor whose liveness check was
`pgrep -f "train.py --config ...<run>"` matched the shell wrapper whose command line contained that
pattern, concluded the trainer was already running, and slept forever without launching anything.
The same trap made a `pkill -f` kill its own shell. Liveness is now a PID file plus `kill -0`, and
nothing here matches on a command-line pattern.

**A completion test read the wrong source.** The NaN tracer logs every 10 steps, so it prints
46,870 and never 46,875. A chain waiting on `tracer_step >= target` hung forever and left the GPU
idle 53 h. Completion is now tested on the checkpoint's `global_step`, which is authoritative.

## Why verify_s3.sh exists

A truncated upload ETag-verifies perfectly: the ETag proves the transfer was faithful, not that the
file was complete. A disk-full crash once left a 289 MB fragment of a 1,523 MB checkpoint and the
janitor marked it verified. A torch checkpoint is a zip, so this script reads the real tail back
from S3 and looks for the end-of-central-directory record, which a truncated file lacks.
`ckpt_janitor.sh` additionally refuses any checkpoint under 1 GB outright.
