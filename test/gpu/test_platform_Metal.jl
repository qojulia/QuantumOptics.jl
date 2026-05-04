@testitem "Metal" tags = [:metal] begin

    include("implementation/test_platform.jl")

    using Metal: MtlArray, Metal
    const AT = MtlArray

    const can_run = Metal.functional()

    @testset "Device availability" begin
        @test can_run
    end

    if can_run
        synchronize() = Metal.synchronize(Metal.global_queue(Metal.device()))
        test_platform(AT, synchronize;
            storage_eltype=ComplexF32,
            time_eltype=Float32,
            solver_kwargs=(reltol=1f-6, abstol=1f-8, adaptive=false, dt=0.1f0)
        )
    else
        @info "Skipping Metal tests - no devices available"
    end

end
