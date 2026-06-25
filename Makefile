PYTHON ?= python3
NPM ?= npm

.PHONY: release web-build

web-build:
	$(NPM) --prefix web run build

release: web-build
	$(PYTHON) tools/build_release.py --output-dir dist
