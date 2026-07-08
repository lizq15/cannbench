PYTHON ?= python3
NPM ?= npm

WEB_NODE_MODULES_STAMP := web/node_modules/.installed

.PHONY: release web-build

$(WEB_NODE_MODULES_STAMP): web/package.json web/package-lock.json
	$(NPM) --prefix web ci
	@touch $@

web-build: $(WEB_NODE_MODULES_STAMP)
	$(NPM) --prefix web run build

release: web-build
	$(PYTHON) tools/build_release.py --output-dir dist
