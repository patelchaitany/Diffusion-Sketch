.PHONY: install test train clean help

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Run setup.sh (create venv, install deps, run tests)
	bash setup.sh

train:  ## Start training with default config
	$(PYTHON) -m diffusion_sketch

clean:  ## Remove generated artifacts
	rm -rf .venv __pycache__ *.egg-info ray_results
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
