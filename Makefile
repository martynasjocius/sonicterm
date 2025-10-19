VENV ?= venv
PYTHON := $(VENV)/bin/python

.PHONY: test tests

test tests:
	$(PYTHON) -m unittest discover -s tests -t .
