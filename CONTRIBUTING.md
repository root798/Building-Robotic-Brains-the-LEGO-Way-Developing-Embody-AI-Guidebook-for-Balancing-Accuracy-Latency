# Contributing

This guide is maintained by [root798](https://github.com/root798). Contributions
should improve reproducibility or document an evidenced failure, not merely add
another unqualified performance number.

## Before proposing a change

1. Keep one coherent change per pull request. Do not rewrite historical results
   to make a new experiment look better.
2. Identify the model, domain, split, initialization, seed, effective batch size,
   precision, schedule and evaluation settings.
3. Record whether the evidence is a synthetic regression, a CPU integration
   test, a full training run, or a checkpoint-bound evaluation. These are not
   interchangeable levels of validation.
4. For a status update, include its UTC timestamp and preserve the distinction
   between completion, finite-loss health, checkpoint verification and accuracy.
   A candidate replacement is a revision of a job, not an additional model.
5. Never commit datasets, private annotations, checkpoints, credentials, machine
   paths, process IDs, raw private logs or other people's files. Use explicit
   staging paths and review the entire staged diff.

## Validation

The standard-library publication checks require no installation or GPU:

```bash
python -m unittest discover -s tests -p 'test_training_configs.py'
python -m unittest discover -s tests -p 'test_publication.py'
python -m unittest discover -s tests -p 'test_train_cli.py'
git diff --check
```

For changes to the numerical guard, also run the CPU hook regression in the
documented training environment. CUDA regression is a separate, explicitly
scheduled check; do not use a busy shared GPU. Full model training and dataset
evaluation are not performed by the lightweight publication CI.

## Results and corrections

Never silently replace a checkpoint or reuse an evaluation from a different
checkpoint. Keep old evidence and publish a new run identity. Mark historical
or failed rows, describe what changed, and require a new evaluation before a
repair is called complete. Fixing a loss kernel does not prove that all other
parts of a training recipe are correct.

Preserve upstream attribution and license notices. Dataset licenses and model
weight redistribution rights must be checked separately before proposing assets.
This update does not select a blanket license for the repository.
