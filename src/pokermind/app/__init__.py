"""M8A: local application backend for playing Leduc against the frozen adaptive AI.

* ``context``: loads and verifies the frozen research baseline.
* ``session``: hand engine, hidden-information-safe views, research metrics.
* ``policy_update`` and ``updates``: background RNR recomputation with
  certified, atomic installation.
* ``manager``: the session store.
* ``api``: a thin FastAPI adapter (optional ``[app]`` extra).
"""
