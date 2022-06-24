# client execution by make
all: pre_inst cln_cah run

pre_inst:
	pip install -e .
	pip install -r requirements.txt

make_req:
	pip list --format=freeze > requirements.txt

# The running cmd is described in sh script
run: cln_cah
	export CONFIGER_PATH="./run_script/pretrain/xxx.ini"; \
	python3 linear_protocol.py

#.PHONY: test  # havn't build the test yet..
#test:
#    python test/*.py

cln_cah:
	find . -type f -name *.pyc -delete
	find . -type d -name __pycache__ -delete

cln_ckpt: 
	find . -type f -name *.ckpt -delete
	
.PHONY: clean
clean: cln_cah cln_ckpt
	rm ./meta_info/logs/* -rf