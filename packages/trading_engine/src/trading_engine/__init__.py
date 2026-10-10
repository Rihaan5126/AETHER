"""AETHER research engine.

The engine is a pure-Python library with no web or database dependencies:

- ``data``: market data providers, validation and immutable dataset versions
- ``features``: point-in-time feature pipeline
- ``training``: labels, chronological splits and walk-forward folds
- ``agents``: baseline and ML trading agents behind one interface
- ``risk``: centralised risk controls applied by the simulator
- ``simulation``: deterministic event-driven long-only portfolio simulator
- ``evaluation``: performance metrics and statistical warnings
- ``tournaments``: fair multi-agent evaluation and ranking
- ``evolution``: typed genomes, mutation, selection and lineage
- ``experiments``: experiment configuration, manifests and orchestration

All simulated money is fictional. Nothing in this package places real orders.
"""

__version__ = "0.1.0"
