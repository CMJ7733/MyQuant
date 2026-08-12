"""Live monitoring for a running (or finished) search.

Design constraint that shapes everything here: **the monitor never imports the
search.**  It reads the two JSONL streams the run already writes —
``generations.jsonl`` and ``llm_calls.jsonl`` — and nothing else.  Consequences,
all of them wanted:

* zero overhead and zero risk for the search itself; a crash in the monitor cannot
  take down a six-hour run, and vice versa;
* the same code monitors a live run and replays a finished one, so a post-mortem
  uses the identical view;
* you can attach and detach at any time, as many times as you like.

The two files are append-only and flushed per record, so tailing them by byte
offset is both cheap and correct.

What the UI can show, and what it cannot
----------------------------------------
Agents run **sequentially** (``loop.py``: ``for agent in agents``), so at any
instant exactly one of the 21 is working.  The dashboard therefore shows one active
agent, the finished ones with their results, and the queued ones greyed out — plus
the eight agents the golden-ratio selection did not pick this run, which is the
paper's design (§B.8) and not a fault.
"""

from cogalpha.monitor.reader import RunReader, RunState  # noqa: F401

__all__ = ["RunReader", "RunState"]
