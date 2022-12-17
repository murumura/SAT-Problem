docker_dir := ./scripts
src_dir := ./internal
src_files := $(wildcard	$(src_dir)/*.py *.py) 
test_dir := ./tests
test_files := $(wildcard $(test_dir)/*.py) 
.PHONY: docker_run
docker_run: 
	-@sh $(docker_dir)/docker_run.sh

.PHONY: format
format:
	@echo "Format: " $(src_files)
	yapf -i $(src_files) $(test_files)

.PHONY: clean
clean:
	@echo "Clean CKPT dir: ./checkpoint"
	rm -rf ./checkpoint
	@echo "Clean PYC file:"
	find . -name "*.pyc" -type f -delete
	@echo "Clean PYC folder:"
	rm -rf ./internal/__pycache__
	rm -rf ./__pycache__ ./tests/__pycache__
	@echo "Clean ipynb checkpoint:"
	rm -rf .ipynb_checkpoints ./internal/.ipynb_checkpoints 
