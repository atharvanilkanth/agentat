.PHONY: install run test clean

install:
	pip install -e .

run:
	python scripts/run_all.py run-all

test:
	pytest tests/ -v

clean:
	rm -rf outputs/*.pkl outputs/*.json outputs/*.parquet outputs/*.csv outputs/qualitative_examples/
