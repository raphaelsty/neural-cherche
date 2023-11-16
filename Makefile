COMMIT_HASH := $(shell eval git rev-parse HEAD)

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/*/*.ipynb --ExecutePreprocessor.timeout=-1

render-notebooks:

livedoc:
	mkdocs build --clean
	mkdocs serve --dirtyreload

deploydoc:
	mkdocs gh-deploy --force

.PHONY: bench
bench:
	asv run ${COMMIT_HASH} --config benchmarks/asv.conf.json --steps 1
	asv run master --config benchmarks/asv.conf.json --steps 1
	asv compare the-merge ${COMMIT_HASH} --config benchmarks/asv.conf.json