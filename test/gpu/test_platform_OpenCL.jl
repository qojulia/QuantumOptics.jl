@testitem "OpenCL" tags = [:opencl] begin

    include("implementation/test_platform.jl")

    using OpenCL: CLArray, cl
    const AT = CLArray

    const can_run = try
        length(cl.platforms()) > 0 && any(
            length(cl.devices(platform)) > 0 for platform in cl.platforms()
        )
    catch
        false
    end

    @testset "Device availability" begin
        @test can_run
    end

    if can_run
        synchronize() = cl.finish(cl.queue())
        test_platform(AT, synchronize)
    else
        @info "Skipping OpenCL tests - no devices available"
    end

end