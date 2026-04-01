.PHONY: install test test-data test-config test-transforms train tensorboard clean help

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Run setup.sh (create venv, install deps, run tests)
	bash setup.sh

test:  ## Run all tests
	$(PYTEST) tests/ -v

test-data:  ## Run dataset preparation tests
	$(PYTEST) tests/test_dataset.py -v

test-transforms:  ## Run transform tests
	$(PYTEST) tests/test_transforms.py -v

test-config:  ## Run config loading tests
	$(PYTEST) tests/test_config.py -v

train:  ## Start training with default config
	$(PYTHON) -m diffusion_sketch

tensorboard:  ## Launch TensorBoard to view training metrics
	$(PYTHON) -m tensorboard.main --logdir runs --bind_all

clean:  ## Remove generated artifacts
	rm -rf .venv __pycache__ *.egg-info ray_results
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
