# GPU test flags
GPU_TEST = "false"

if Sys.iswindows()
    @info "Skipping GPU tests -- only executed on *NIX platforms."
else
    GPU_TEST = lowercase(get(ENV, "GPU_TEST", "false"))
    if !(GPU_TEST in ("false", "cuda", "amdgpu", "opencl", "metal"))
        error("GPU_TEST must be one of: false, cuda, amdgpu, opencl, metal")
    end

    if GPU_TEST == "false"
        @info "Skipping GPU tests -- must be explicitly enabled."
        @info "Environment must set GPU_TEST to one of: cuda, amdgpu, opencl, metal."
    else
        @info "Running with $(GPU_TEST) tests."
    end
end

CUDA_flag = GPU_TEST == "cuda"
AMDGPU_flag = GPU_TEST == "amdgpu"
OpenCL_flag = GPU_TEST == "opencl"
Metal_flag = GPU_TEST == "metal"

using Pkg
CUDA_flag && Pkg.add("CUDA")
AMDGPU_flag && Pkg.add("AMDGPU")
OpenCL_flag && Pkg.add(["pocl_jll", "OpenCL"])
Metal_flag && Pkg.add("Metal")
if any((CUDA_flag, AMDGPU_flag, OpenCL_flag, Metal_flag))
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

  if Metal_flag
    return :metal in ti.tags
  else
    push!(exclude, :metal)
  end

  if !(VERSION >= v"1.10")
    push!(exclude, :aqua)
  end

  return all(!in(exclude), ti.tags)
end

println("Starting tests with $(Threads.nthreads()) threads out of `Sys.CPU_THREADS = $(Sys.CPU_THREADS)`...")

@run_package_tests filter=testfilter
