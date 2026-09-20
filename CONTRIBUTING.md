# Contributing

Contributions to the model zoo, recipes, evaluation tools, and documentation
are welcome. The repository is maintained by [root798](https://github.com/root798).

## Adding a model or result

1. Include the model, domain, configuration, initialization, seed, batch size,
   schedule, and framework versions.
2. Evaluate the complete split using the [shared protocol](docs/evaluation.md).
3. Record checkpoint and configuration hashes with the aggregate metrics.
4. Give recipe changes a distinct run identity and update the domain table.

Keep dataset images, private paths, credentials, and raw training logs out of
commits. Weight releases require compatible redistribution terms and a download
location. Preserve upstream authorship and license notices.

## Checks

```bash
python -m unittest discover -s tests -p 'test_training_configs.py'
python -m unittest discover -s tests -p 'test_publication.py'
python -m unittest discover -s tests -p 'test_model_zoo.py'
python -m unittest discover -s tests -p 'test_train_cli.py'
git diff --check
```

Changes to the loss adapter also use the CPU hook tests in the
[reproduction guide](docs/reproduction.md). GPU training and evaluation run
separately from lightweight CI.
