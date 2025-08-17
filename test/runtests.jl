# GPU test flags
CUDA_flag = false
AMDGPU_flag = false
OpenCL_flag = false

if Sys.iswindows()
    @info "Skipping GPU tests -- only executed on *NIX platforms."
else
    CUDA_flag = get(ENV, "CUDA_TEST", "") == "true"
    AMDGPU_flag = get(ENV, "AMDGPU_TEST", "") == "true"
    OpenCL_flag = get(ENV, "OpenCL_TEST", "") == "true"

    CUDA_flag && @info "Running with CUDA tests."
    AMDGPU_flag && @info "Running with AMDGPU tests."
    OpenCL_flag && @info "Running with OpenCL tests."
    if !any((CUDA_flag, AMDGPU_flag, OpenCL_flag))
        @info "Skipping GPU tests -- must be explicitly enabled."
        @info "Environment must set [CUDA, AMDGPU, OpenCL]_TEST=true."
    end
end

using Pkg
CUDA_flag && Pkg.add("CUDA")
AMDGPU_flag && Pkg.add("AMDGPU")
OpenCL_flag && Pkg.add("OpenCL")
if any((CUDA_flag, AMDGPU_flag, OpenCL_flag))
    Pkg.add("Adapt")
end

using TestItemRunner
using QuantumOptics

# filter for the test
testfilter = ti -> begin
  exclude = Symbol[:slow]
  
  if get(ENV,"JET_TEST","")=="true"
    return :jet in ti.tags
  else
    push!(exclude, :jet)
  end
  
  if CUDA_flag
    return :cuda in ti.tags
  else
    push!(exclude, :cuda)
  end

  if AMDGPU_flag
    return :amdgpu in ti.tags
  else
    push!(exclude, :amdgpu)
  end

  if OpenCL_flag
    return :opencl in ti.tags
  else
    push!(exclude, :opencl)
  end
  
  if !(VERSION >= v"1.10")
    push!(exclude, :aqua)
  end

  return all(!in(exclude), ti.tags)
end

println("Starting tests with $(Threads.nthreads()) threads out of `Sys.CPU_THREADS = $(Sys.CPU_THREADS)`...")

@run_package_tests filter=testfilter