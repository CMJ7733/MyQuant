"""Live monitoring for a running (or finished) Famou experiment.

Design constraint that shapes everything here: **the monitor never imports the
evolution framework.**  It reads the artifacts a run already writes to its
experiment directory -- ``results/``, ``programs/``, ``experiment_checkpoint_*.json``,
``config.yaml``, ``experiment.jsonl``, ``llm_requests.log`` -- and nothing else.
Consequences, all of them wanted:

* zero overhead and zero risk for the evolution itself; a crash in the monitor cannot
  take down a six-hour run, and vice versa;
* the same code monitors a live run and replays a finished one, so a post-mortem
  uses the identical view;
* you can attach and detach at any time, as many times as you like.

Relationship to the other two UIs
---------------------------------
``api_server/fm_api.py`` is the *control* plane (``/init``, ``/evolve/start`` ...) and
``api_server/dashboard.html`` is the *post-mortem* report ``report.py`` fills in with
an LLM once a run has finished.  Neither shows anything while evolution is running.
This package is that missing third view, and it does not replace either of them.

Optional control
----------------
When started with ``--api-base``, the server proxies pause/continue/stop through to
the existing ``api_server`` endpoints.  Without it the dashboard is strictly
read-only and needs no other process running.
"""

from famou.monitor.reader import ExperimentReader, RunState  # noqa: F401

__all__ = ["ExperimentReader", "RunState"]
