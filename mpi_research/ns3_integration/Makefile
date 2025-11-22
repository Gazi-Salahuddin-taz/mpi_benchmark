# NS-3 Integration Makefile
PROJECT_ROOT := $(shell pwd)
NS3_DIR ?= $(HOME)/wangyl/ns-allinone-3.37/ns-3.37
SCENARIOS_DIR := $(PROJECT_ROOT)/scenarios
SCRATCH_DIR := $(NS3_DIR)/scratch
RESULTS_DIR := $(PROJECT_ROOT)/results

.PHONY: all build clean setup run-fat-tree run-torus validate

all: setup build

setup:
	@echo "Setting up scenarios..."
	@cp $(SCENARIOS_DIR)/*.cc $(SCRATCH_DIR)/
	@echo "Scenarios copied to NS-3 scratch directory"

build:
	@echo "Building NS-3 scenarios..."
	@cd $(NS3_DIR) && ./ns3 build

clean:
	@echo "Cleaning build..."
	@cd $(NS3_DIR) && ./ns3 clean
	@rm -rf $(RESULTS_DIR)/*

run-fat-tree: setup build
	@echo "Running Fat Tree simulation..."
	@cd $(NS3_DIR) && ./ns3 run "fat_tree_scenario --k=4 --duration=10"

run-torus: setup build
	@echo "Running Torus simulation..."
	@cd $(NS3_DIR) && ./ns3 run "torus_scenario --x=4 --y=4 --duration=10"

validate:
	@echo "Validating setup..."
	@test -d $(NS3_DIR) || (echo "NS3_DIR not found: $(NS3_DIR)" && exit 1)
	@test -f $(NS3_DIR)/ns3 || (echo "ns3 executable not found" && exit 1)
	@test -d $(SCENARIOS_DIR) || (echo "Scenarios directory not found: $(SCENARIOS_DIR)" && exit 1)
	@echo "Setup validation passed"

help:
	@echo "Available targets:"
	@echo "  all        - Setup and build everything"
	@echo "  setup      - Copy scenarios to NS-3 scratch"
	@echo "  build      - Build NS-3 scenarios"
	@echo "  clean      - Clean build and results"
	@echo "  run-fat-tree - Run fat tree simulation"
	@echo "  run-torus  - Run torus simulation"
	@echo "  validate   - Validate setup"
	@echo "  help       - Show this help"