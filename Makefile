.PHONY: install run prototype test clean

install:
	pip install -e .

run: prototype

prototype:
	python scripts/run_all.py run-all

test:
	pytest tests/ -v

ablation:
	python scripts/ablation.py

clean:
	rm -rf outputs/*.pkl outputs/*.json outputs/*.parquet outputs/*.csv outputs/qualitative_examples/

