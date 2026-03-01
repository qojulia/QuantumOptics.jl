# Makefile for running comparative benchmarks and updating the benchmark webpage.
#
# Usage:
#   make all             - Run all benchmarks, collect results
#   make julia           - Run only QuantumOptics.jl benchmarks
#   make qutip           - Run only QuTiP benchmarks
#   make quantumtoolbox  - Run only QuantumToolbox.jl benchmarks
#   make collect         - Collect results into consolidated JSON files
#   make webpage         - Copy results to the website directory
#   make clean           - Remove all generated results
#
# The WEBSITE variable should point to the QuantumOptics.jl-website checkout:
#   make webpage WEBSITE=../../QuantumOptics.jl-website

JULIA ?= julia
PYTHON ?= python3
WEBSITE ?= ../../../QuantumOptics.jl-website

.PHONY: all julia qutip quantumtoolbox collect webpage clean specs help

help:
	@echo "Comparative Benchmark Suite for QuantumOptics.jl"
	@echo ""
	@echo "Targets:"
	@echo "  all             - Run all benchmarks and collect results"
	@echo "  julia           - Run QuantumOptics.jl benchmarks"
	@echo "  qutip           - Run QuTiP (Python) benchmarks"
	@echo "  quantumtoolbox  - Run QuantumToolbox.jl benchmarks"
	@echo "  collect         - Collect results into consolidated JSON"
	@echo "  webpage         - Copy results to the website directory"
	@echo "  clean           - Remove generated results"
	@echo ""
	@echo "Variables:"
	@echo "  JULIA=julia     - Path to Julia executable"
	@echo "  PYTHON=python3  - Path to Python executable"
	@echo "  WEBSITE=path    - Path to QuantumOptics.jl-website checkout"

all: specs julia qutip quantumtoolbox collect

specs:
	@mkdir -p results results-collected
	$(PYTHON) hardware_specs.py

julia:
	@mkdir -p results
	@echo "Running QuantumOptics.jl benchmarks..."
	@cd benchmarks-QuantumOptics.jl && \
	for f in $$(ls *.jl | grep -v benchmarkutils); do \
		echo "  $$f"; \
		$(JULIA) --project=../../.. $$f; \
	done

qutip:
	@mkdir -p results
	@echo "Running QuTiP benchmarks..."
	@cd benchmarks-QuTiP && \
	for f in $$(ls *.py | grep -v benchmarkutils | grep -v __init__); do \
		echo "  $$f"; \
		$(PYTHON) $$f; \
	done

quantumtoolbox:
	@mkdir -p results
	@echo "Running QuantumToolbox.jl benchmarks..."
	@cd benchmarks-QuantumToolbox.jl && \
	for f in $$(ls *.jl | grep -v benchmarkutils); do \
		echo "  $$f"; \
		$(JULIA) $$f; \
	done

collect:
	@mkdir -p results-collected
	$(PYTHON) collect_results.py

webpage: collect
	@if [ ! -d "$(WEBSITE)" ]; then \
		echo "Error: Website directory not found at $(WEBSITE)"; \
		echo "Set WEBSITE variable: make webpage WEBSITE=/path/to/QuantumOptics.jl-website"; \
		exit 1; \
	fi
	@mkdir -p "$(WEBSITE)/src/benchmark-data"
	cp -r results-collected/* "$(WEBSITE)/src/benchmark-data/"
	@echo "Benchmark data copied to $(WEBSITE)/src/benchmark-data/"

clean:
	rm -f results/*.json
	rm -f results-collected/*.json
	rm -rf sourcecode/
	rm -rf __pycache__/
	rm -rf benchmarks-QuTiP/__pycache__/
