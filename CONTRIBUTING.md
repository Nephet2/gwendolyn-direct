# Contributing

Bug reports should include the Gwendolyn version, Windows version, model server, model name and the output of `CHECK_SETUP.bat`. Remove private text and local paths before posting.

For code changes, run `python -m unittest discover -p "test_*.py"` and `python check_setup.py --offline` before opening a pull request. Do not commit generated configuration, memory, logs, voice recordings or model files.
