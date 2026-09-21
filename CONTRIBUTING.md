# Contributing

Contributions that improve retrieval quality, grounding, evaluation rigor,
accessibility or documentation are welcome.

## Development workflow

1. Fork the repository and create a focused branch.
2. Create a Python 3.12 virtual environment.
3. Install `requirements-runtime.txt`.
4. Make the smallest coherent change.
5. Add or update tests.
6. Run the full suite before opening a pull request.

```powershell
python -m unittest discover -s tests -p "test_*.py" -q
```

## Pull-request expectations

- explain the problem and the chosen solution;
- report tests and any paid evaluation cost;
- do not silently change frozen prompts or runtime artifacts;
- include before/after metrics for retrieval or generation changes;
- preserve citation validation, abstention and cost controls;
- update `MODEL_CARD.md` when capabilities or limitations change.

## Data and secrets

Never commit `.env` files, API keys, raw datasets, generated caches, personal
annotation exports or user conversations. Use `.env.example` for configuration
names and synthetic data for tests.

