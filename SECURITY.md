# Security and private data

Gwendolyn is intended to run on the same computer as its model server and Vector. Keep every local service bound to `127.0.0.1` unless you understand and secure the network exposure.

The following files may contain private conversation, preferences or machine-specific paths and are ignored by Git: `config.json`, `director_profile.json`, `director_voices.json`, `private_lexicon.json`, `gwendolyn_memory.json`, logs, session journals, handoffs and voice recordings.

Before sharing a diagnostic bundle, inspect it yourself. Report a security issue privately through GitHub's security advisory feature when available.
