docker_dir := ./scripts
src_dir := ./internal

.PHONY: docker_run
docker_run: 
	-@sh $(docker_dir)/docker_run.sh
