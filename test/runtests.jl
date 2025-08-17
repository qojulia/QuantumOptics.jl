using TestItemRunner
using QuantumOptics

# Define test filters
testfilter = ti->(!(:slow in ti.tags))

# GPU test filters
gpu_test_filter = ti -> begin
    cuda_test = get(ENV, "CUDA_TEST", "false") == "true"
    amdgpu_test = get(ENV, "AMDGPU_TEST", "false") == "true" 
    opencl_test = get(ENV, "OpenCL_TEST", "true") == "true"  # Default to true for CI
    
    # Include GPU tests based on environment variables
    if :cuda in ti.tags
        return cuda_test
    elseif :amdgpu in ti.tags
        return amdgpu_test
    elseif :opencl in ti.tags
        return opencl_test
    else
        return !(:slow in ti.tags) && !(:cuda in ti.tags) && !(:amdgpu in ti.tags) && !(:opencl in ti.tags)
    end
end

@run_package_tests filter=gpu_test_filter