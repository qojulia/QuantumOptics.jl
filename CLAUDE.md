# QuantumOptics.jl

QuantumOptics.jl is a numerical framework written in Julia that makes it easy to simulate various kinds of quantum systems. It builds on QuantumOpticsBase.jl and provides dynamic solvers for time evolution, master equations, stochastic processes, and more.

## Project Structure

- `src/` - Main source code
  - `QuantumOptics.jl` - Main module file
  - `schroedinger.jl` - Schrödinger equation time evolution
  - `master.jl` - Master equation dynamics
  - `mcwf.jl` - Monte Carlo wave function methods
  - `stochastic_*.jl` - Stochastic differential equation solvers
  - `semiclassical.jl` - Semiclassical dynamics
  - `steadystate.jl` - Steady state calculations
  - `timecorrelations.jl` - Time correlation functions
  - `phasespace.jl` - Phase space calculations
  - `spectralanalysis.jl` - Spectral analysis tools
- `test/` - Comprehensive test suite
- `benchmark/` - Performance benchmarking

## Development Commands

### Running Tests
```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run with only specific GPU backend tests
CUDA_TEST=true julia --project=. -e "using Pkg; Pkg.test()"
AMDGPU_TEST=true julia --project=. -e "using Pkg; Pkg.test()"
OpenCL_TEST=true julia --project=. -e "using Pkg; Pkg.test()"

# Run specific test files
julia --project=. -e "using TestItemRunner; @run_package_tests filter=ti->contains(string(ti.name), \"schroedinger\")"
```

### Building Documentation
```bash
# Build documentation (handled by the main QuantumOptics.jl-documentation repository)
# See: https://github.com/qojulia/QuantumOptics.jl-documentation
```

### Package Management
```bash
# Instantiate project dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Update dependencies
julia --project=. -e "using Pkg; Pkg.update()"

# Check package status
julia --project=. -e "using Pkg; Pkg.status()"
```

### Benchmarking
```bash
# Run benchmarks
julia --project=benchmark benchmark/benchmarks.jl
```

## Testing Information

The test suite uses TestItemRunner and includes:

- Time evolution tests (Schrödinger, master equation, MCWF)
- Stochastic differential equation tests
- Semiclassical dynamics tests
- Steady state calculation tests
- Time correlation function tests
- Phase space and spectral analysis tests
- Integration tests with ForwardDiff.jl for automatic differentiation
- Code quality tests (Aqua.jl, JET.jl)

Special test configurations:
- JET tests run when `JET_TEST=true` environment variable is set
- GPU tests run when `CUDA_TEST=true`, `AMDGPU_TEST=true`, or `OpenCL_TEST=true` are set

## Key Dependencies

- `QuantumOpticsBase.jl` - Provides fundamental types and operations
- `OrdinaryDiffEqCore.jl` - Core ODE solver functionality
- `OrdinaryDiffEqLowOrderRK.jl` - Low-order Runge-Kutta methods
- `StochasticDiffEq.jl` - Stochastic differential equation solvers
- `DiffEqCallbacks.jl` - Callback functionality for differential equations
- `LinearMaps.jl` - Linear map abstractions
- `KrylovKit.jl` - Krylov subspace methods
- `IterativeSolvers.jl` - Iterative linear algebra solvers
- `Arpack.jl` - Eigenvalue problems

## Development Notes

- Minimum Julia version: 1.10
- Uses semantic versioning
- Extensive test coverage with multiple CI platforms
- Documentation hosted at https://docs.qojulia.org/
- Compatible with GPU acceleration (CUDA, AMDGPU, OpenCL)
- Supports automatic differentiation via ForwardDiff.jl

## GPU Support

QuantumOptics.jl supports GPU acceleration through:
- CUDA.jl for NVIDIA GPUs
- AMDGPU.jl for AMD GPUs  
- OpenCL.jl for OpenCL-compatible devices

GPU tests are located in `test/gpu/` and follow the same structure as QuantumOpticsBase.jl.

## Related Packages

- `QuantumOpticsBase.jl` - Base functionality and types
- `QuantumInterface.jl` - Common quantum computing interfaces
- See the @qojulia organization for the full ecosystem

## Code Formatting

### Removing Trailing Whitespaces
Before committing, ensure there are no trailing whitespaces in Julia files:

```bash
# Remove trailing whitespaces from all .jl files (requires gnu tools)
find . -type f -name '*.jl' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
```

### Ensuring Files End with Newlines
Ensure all Julia files end with a newline to avoid misbehaving CLI tools:

```bash
# Add newline to end of all .jl files that don't have one
find . -type f -name '*.jl' -exec sed -i '$a\' {} \+
```

### General Formatting Guidelines
- Use 4 spaces for indentation (no tabs)
- Remove trailing whitespaces from all lines
- Ensure files end with a single newline
- Follow Julia standard naming conventions
- Keep lines under 100 characters when reasonable

## Contributing

This package follows standard Julia development practices:
- **Always pull latest changes first**: Before creating any new feature or starting work, ensure you have the latest version by running `git pull origin master` (or `git pull origin main`)
- **Pull before continuing work**: Other maintainers might have modified the branch you are working on. Always call `git pull` before continuing work on an existing branch
- **Push changes to remote**: Always push your local changes to the remote branch to keep the PR up to date: `git push origin <branch-name>`
- **Run all tests before submitting**: Before creating or updating a PR, always run the full test suite to ensure nothing is broken: `julia --project=. -e "using Pkg; Pkg.test()"`
- Fork and create feature branches
- Write tests for new functionality
- Ensure all tests pass before merging
- **Keep PRs focused**: A PR should implement one self-contained change. Avoid mixing feature work with formatting changes to unrelated files, even for improvements like adding missing newlines. Format unrelated files in separate commits or PRs.