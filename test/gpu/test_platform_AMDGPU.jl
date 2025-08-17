@testitem "AMDGPU" tags = [:amdgpu] begin

    include("implementation/test_platform.jl")

    using AMDGPU: ROCArray, AMDGPU
    const AT = ROCArray

    const can_run = AMDGPU.functional()

    @testset "Device availability" begin
        @test can_run
    end

    if can_run
        synchronize() = AMDGPU.synchronize()
        test_platform(AT, synchronize)
    else
        @info "Skipping AMDGPU tests - no devices available"
    end

end