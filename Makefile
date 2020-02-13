all:
	python3 -m pylint ngboost
pkg:
	python3 setup.py sdist bdist_wheel
clean:
	rm -r build dist ngboost.egg-info
upload:
	twine upload dist/*

build:
	@docker build -t ngboost .

run:
	@docker run \
		--rm \
		--name ngboost \
		-e PYTHONDONTWRITEBYTECODE=1 \
		-v $(PWD):/src \
		ngboost $(file)

test:
	@make -s run file='-m pytest -p no:warnings'
