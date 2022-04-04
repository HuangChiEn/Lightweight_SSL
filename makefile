# python makefile

all: clean

# The running cmd is described in sh script
run:
	run_script/run_pretrain.sh
	run_script/run_eval.sh

#.PHONY: test
#test:
#    python test/*.py

#.PHONY: release
#release:
#    python3 setup.py sdist bdist_wheel upload

.PHONY: clean
clean:
    find . -type f -name *.pyc -delete
    find . -type d -name __pycache__ -delete