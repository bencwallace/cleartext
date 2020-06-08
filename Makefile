.PHONY: init spacy

init:
	pip install -e ./
	python -m spacy download en_core_web_sm
