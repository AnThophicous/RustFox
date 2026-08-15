BUILD   ?= build
TYPE    ?= Release
JOBS    ?= $(shell nproc 2>/dev/null || echo 2)
CMAKE   ?= cmake

.PHONY: all configure build test probe watch clean docker docker-run format-check

all: build

configure:
	$(CMAKE) -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=$(TYPE) -DFOX_WERROR=ON

build: configure
	$(CMAKE) --build $(BUILD) -j $(JOBS)

test: build
	cd $(BUILD) && ctest --output-on-failure

probe: build
	./$(BUILD)/fox probe --bench

watch: build
	./$(BUILD)/fox watch --seconds 20

clean:
	rm -rf $(BUILD)

docker:
	docker build -f docker/Dockerfile -t rustfox:dev .

docker-run: docker
	docker run --rm -it --memory=1g --cpus=2 rustfox:dev probe --bench
