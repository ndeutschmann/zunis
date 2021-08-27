.PHONY: clean data requirements doc static-doc docinstall serve build pypi-upload

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = zunis
PYTHON_INTERPRETER = python
DOC_DIR = docs/
DOC_BUILD_DIR = doc_build/
PROJECT_SRC = zunis_lib/

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf dist zunis.egg-info

## Build automatic API documentation
doc:
	$(MAKE) clean -C $(DOC_BUILD_DIR)
	$(MAKE) cleanapi -C $(DOC_BUILD_DIR)
	$(MAKE) apidoc -C $(DOC_BUILD_DIR)
	$(MAKE) html -C $(DOC_BUILD_DIR)

## Build documentation from the current state of ./docs
static-doc:
	$(MAKE) clean -C $(DOC_BUILD_DIR)
	$(MAKE) html -C $(DOC_BUILD_DIR)

## Install the built documentation
doc-install:
	rm -rf $(DOC_DIR)
	cp -R $(DOC_BUILD_DIR)/_build/html/ $(DOC_DIR)
	touch $(DOC_DIR)/.nojekyll

serve:
	$(MAKE) serve -C $(DOC_BUILD_DIR)

build:
	python setup.py sdist bdist_wheel

pypi-upload:
	python -m twine upload dist/*


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
