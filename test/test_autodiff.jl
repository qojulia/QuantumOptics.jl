@testitem "Test automatic differentation backends with QuantumOptics.jl" begin
    using Test
    using QuantumOptics
    using DifferentiationInterface
    import ForwardDiff
    import FiniteDiff
    
    # cost parameters
    pnum, pvec = rand(Float64), rand(Float64, 2)
    # in-place buffers
    bufvec, bufmat = similar(pvec), similar(pvec * pvec')
    # random system definitions
    basis = SpinBasis(10//1)
    ψ0 = randstate(basis)
    ρ0 = dm(ψ0)
    H1, H2 = randoperator(basis), randoperator(basis)
    H1t, H2t = TimeDependentSum(sin=>H1), TimeDependentSum(sin=>H2)
    tspan = [0.0, 1.0]

    @testset "time-evolution" begin

        @testset "schroedinger" begin

            function cost(p::Real)
                _, ψt = timeevolution.schroedinger(tspan, ψ0, p*H1)
                return 1 - norm(ψt)^2
            end
            function costvec(p::Real)
                _, ψt = timeevolution.schroedinger(tspan, ψ0, p*H1)
                return [1 - norm(ψt)^2, norm(ψt)]
            end
            function cost(p::Vector)
                _, ψt = timeevolution.schroedinger(tspan, ψ0, p[1]*H1 + p[2]*H2)
                return 1 - norm(ψt)^2
            end
            function costvec(p::Vector)
                _, ψt = timeevolution.schroedinger(tspan, ψ0, p[1]*H1 + p[2]*H2)
                return [1 - norm(ψt)^2, norm(ψt)]
            end
            for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                # out-of-place
                @test_nowarn derivative(cost, backend, pnum)
                @test_nowarn second_derivative(cost, backend, pnum)
                @test_nowarn gradient(cost, backend, pvec)
                @test_nowarn jacobian(costvec, backend, pvec)
                @test_nowarn hessian(cost, backend, pvec)
                # in-place
                @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                @test_nowarn gradient!(cost, bufvec, backend, pvec)
                @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                @test_nowarn hessian!(cost, bufmat, backend, pvec)
            end
        end

        @testset "dynamic schroedinger" begin

            @testset "time-dependent functions" begin

                function cost(p::Real)
                    _, ψt = timeevolution.schroedinger_dynamic(tspan, ψ0, (t, ψ) -> exp(p*t)*H1)
                    return 1 - norm(ψt)^2
                end
                function costvec(p::Real)
                    _, ψt = timeevolution.schroedinger_dynamic(tspan, ψ0, (t, ψ) -> exp(p*t)*H1)
                    return [1 - norm(ψt)^2, norm(ψt)]
                end
                function cost(p::Vector)
                    _, ψt = timeevolution.schroedinger_dynamic(tspan, ψ0, (t, ψ) -> exp(p[1]*t)*H1 + exp(p[2]*t)*H2)
                    return 1 - norm(ψt)^2
                end
                function costvec(p::Vector)
                    _, ψt = timeevolution.schroedinger_dynamic(tspan, ψ0, (t, ψ) -> exp(p[1]*t)*H1 + exp(p[2]*t)*H2)
                    return [1 - norm(ψt)^2, norm(ψt)]
                end
                for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                    # out-of-place
                    @test_nowarn derivative(cost, backend, pnum)
                    @test_nowarn second_derivative(cost, backend, pnum)
                    @test_nowarn gradient(cost, backend, pvec)
                    @test_nowarn jacobian(costvec, backend, pvec)
                    @test_nowarn hessian(cost, backend, pvec)
                    # in-place
                    @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                    @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                    @test_nowarn gradient!(cost, bufvec, backend, pvec)
                    @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                    @test_nowarn hessian!(cost, bufmat, backend, pvec)
                end
            end

            @testset "TimeDependentOperators" begin

                function cost(p::Real)
                    _, ψt = timeevolution.schroedinger_dynamic(tspan, ψ0, p*H1t)
                    return 1 - norm(ψt)^2
                end
                function costvec(p::Real)
                    _, ψt = timeevolution.schroedinger_dynamic(tspan, ψ0, p*H1t)
                    return [1 - norm(ψt)^2, norm(ψt)]
                end
                function cost(p::Vector)
                    _, ψt = timeevolution.schroedinger_dynamic(tspan, ψ0, p[1]*H1t + p[2]*H2t)
                    return 1 - norm(ψt)^2
                end
                function costvec(p::Vector)
                    _, ψt = timeevolution.schroedinger_dynamic(tspan, ψ0, p[1]*H1t + p[2]*H2t)
                    return [1 - norm(ψt)^2, norm(ψt)]
                end
                for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                    # out-of-place
                    @test_nowarn derivative(cost, backend, pnum)
                    @test_nowarn second_derivative(cost, backend, pnum)
                    @test_nowarn gradient(cost, backend, pvec)
                    @test_nowarn jacobian(costvec, backend, pvec)
                    @test_nowarn hessian(cost, backend, pvec)
                    # in-place
                    @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                    @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                    @test_nowarn gradient!(cost, bufvec, backend, pvec)
                    @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                    @test_nowarn hessian!(cost, bufmat, backend, pvec)
                end
            end
        end

        for eq in [:master, :master_h, :master_nh]

            @testset "$eq" begin

                # differentiate Hamiltonian
                @testset "Hamiltonian" begin

                    function cost(p::Real)
                         _, ρt = @eval(timeevolution.$eq)(tspan, ρ0, p*H1, [])
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Real)
                        _, ψt = @eval(timeevolution.$eq)(tspan, ρ0, p*H1, [])
                        return [1 - norm(ψt)^2, norm(ψt)]
                    end
                    function cost(p::Vector)
                        _, ρt = @eval(timeevolution.$eq)(tspan, ρ0, p[1]*H1 + p[2]*H2, [])
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Vector)
                        _, ρt = @eval(timeevolution.$eq)(tspan, ρ0, p[1]*H1 + p[2]*H2, [])
                        return [1 - norm(ρt)^2, norm(ρt)]
                    end
                    for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                        # out-of-place
                        @test_nowarn derivative(cost, backend, pnum)
                        @test_nowarn second_derivative(cost, backend, pnum)
                        @test_nowarn gradient(cost, backend, pvec)
                        @test_nowarn jacobian(costvec, backend, pvec)
                        @test_nowarn hessian(cost, backend, pvec)
                        # in-place
                        @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn gradient!(cost, bufvec, backend, pvec)
                        @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                        @test_nowarn hessian!(cost, bufmat, backend, pvec)
                    end
                end

                # differentiate jump operators
                @testset "jump operators" begin

                    function cost(p::Real)
                        _, ρt = @eval(timeevolution.$eq)(tspan, ρ0, H1, [p*H2])
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Real)
                        _, ψt = @eval(timeevolution.$eq)(tspan, ρ0, H1, [p*H2])
                        return [1 - norm(ψt)^2, norm(ψt)]
                    end
                    function cost(p::Vector)
                        _, ρt = @eval(timeevolution.$eq)(tspan, ρ0, H1, [p[1]*H1, p[2]*H2])
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Vector)
                        _, ρt = @eval(timeevolution.$eq)(tspan, ρ0, H1, [p[1]*H1, p[2]*H2])
                        return [1 - norm(ρt)^2, norm(ρt)]
                    end
                    for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                        # out-of-place
                        @test_nowarn derivative(cost, backend, pnum)
                        @test_nowarn second_derivative(cost, backend, pnum)
                        @test_nowarn gradient(cost, backend, pvec)
                        @test_nowarn jacobian(costvec, backend, pvec)
                        @test_nowarn hessian(cost, backend, pvec)
                        # in-place
                        @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn gradient!(cost, bufvec, backend, pvec)
                        @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                        @test_nowarn hessian!(cost, bufmat, backend, pvec)
                    end
                end
            end
        end

        @testset "master_dynamic" begin

            @testset "time-dependent functions" begin

                # differentiate Hamiltonian
                @testset "Hamiltonian" begin

                    function cost(p::Real)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (exp(p*t)*H1, [], []))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Real)
                        _, ψt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (exp(p*t)*H1, [], []))
                        return [1 - norm(ψt)^2, norm(ψt)]
                    end
                    function cost(p::Vector)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (exp(p[1]*t)*H1 + exp(p[2]*t)*H2, [], []))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Vector)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (exp(p[1]*t)*H1 + exp(p[2]*t)*H2, [], []))
                        return [1 - norm(ρt)^2, norm(ρt)]
                    end
                    for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                        # out-of-place
                        @test_nowarn derivative(cost, backend, pnum)
                        @test_nowarn second_derivative(cost, backend, pnum)
                        @test_nowarn gradient(cost, backend, pvec)
                        @test_nowarn jacobian(costvec, backend, pvec)
                        @test_nowarn hessian(cost, backend, pvec)
                        # in-place
                        @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn gradient!(cost, bufvec, backend, pvec)
                        @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                        @test_nowarn hessian!(cost, bufmat, backend, pvec)
                    end
                end

                # differentiate jump operators
                @testset "jump operators" begin

                    function cost(p::Real)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (H1, [exp(p*t)*H2], [dagger(exp(p*t)*H2)]))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Real)
                        _, ψt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (H1, [exp(p*t)*H2], [dagger(exp(p*t)*H2)]))
                        return [1 - norm(ψt)^2, norm(ψt)]
                    end
                    function cost(p::Vector)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (H1, [exp(p[1]*t)*H1, exp(p[2]*t)*H2], [dagger(exp(p[1]*t)*H1), dagger(exp(p[2]*t)*H2)]))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Vector)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (H1, [exp(p[1]*t)*H1, exp(p[2]*t)*H2], [dagger(exp(p[1]*t)*H1), dagger(exp(p[2]*t)*H2)]))
                        return [1 - norm(ρt)^2, norm(ρt)]
                    end
                    for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                        # out-of-place
                        @test_nowarn derivative(cost, backend, pnum)
                        @test_nowarn second_derivative(cost, backend, pnum)
                        @test_nowarn gradient(cost, backend, pvec)
                        @test_nowarn jacobian(costvec, backend, pvec)
                        @test_nowarn hessian(cost, backend, pvec)
                        # in-place
                        @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn gradient!(cost, bufvec, backend, pvec)
                        @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                        @test_nowarn hessian!(cost, bufmat, backend, pvec)
                    end
                end

                # differentiate rates
                @testset "rates" begin

                    function cost(p::Real)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (H1, [H2], [dagger(H2)], [exp(p*t)]))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Real)
                        _, ψt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (H1, [H2], [dagger(H2)], [exp(p*t)]))
                        return [1 - norm(ψt)^2, norm(ψt)]
                    end
                    function cost(p::Vector)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (H1, [H1, H2], [dagger(H1), dagger(H2)], [exp(p[1]*t), exp(p[2]*t)]))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Vector)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, (t, ρ) -> (H1, [H1, H2], [dagger(H1), dagger(H2)], [exp(p[1]*t), exp(p[2]*t)]))
                        return [1 - norm(ρt)^2, norm(ρt)]
                    end
                    for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                        # out-of-place
                        @test_nowarn derivative(cost, backend, pnum)
                        @test_nowarn second_derivative(cost, backend, pnum)
                        @test_nowarn gradient(cost, backend, pvec)
                        @test_nowarn jacobian(costvec, backend, pvec)
                        @test_nowarn hessian(cost, backend, pvec)
                        # in-place
                        @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn gradient!(cost, bufvec, backend, pvec)
                        @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                        @test_nowarn hessian!(cost, bufmat, backend, pvec)
                    end
                end
            end

            @testset "TimeDependentOperators" begin

                # differentiate Hamiltonian
                @testset "Hamiltonian" begin

                    function cost(p::Real)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, p*H1t, [p*H1t])
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Real)
                        _, ψt = timeevolution.master_dynamic(tspan, ρ0, p*H1t, [p*H1t])
                        return [1 - norm(ψt)^2, norm(ψt)]
                    end
                    function cost(p::Vector)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, p[1]*H1t, [p[2]*H1t])
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Vector)
                        _, ρt = timeevolution.master_dynamic(tspan, ρ0, p[1]*H1t, [p[1]*H1t])
                        return [1 - norm(ρt)^2, norm(ρt)]
                    end
                    for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                        # out-of-place
                        @test_nowarn derivative(cost, backend, pnum)
                        @test_nowarn second_derivative(cost, backend, pnum)
                        @test_nowarn gradient(cost, backend, pvec)
                        @test_nowarn jacobian(costvec, backend, pvec)
                        @test_nowarn hessian(cost, backend, pvec)
                        # in-place
                        @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn gradient!(cost, bufvec, backend, pvec)
                        @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                        @test_nowarn hessian!(cost, bufmat, backend, pvec)
                    end
                end
            end
        end

        @testset "master_nh_dynamic" begin

            @testset "time-dependent functions" begin

                # differentiate Hamiltonian
                @testset "Hamiltonian" begin

                    function cost(p::Real)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (exp(p*t)*H1, dagger(exp(p*t)*H1), [], []))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Real)
                        _, ψt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (exp(p*t)*H1, dagger(exp(p*t)*H1), [], []))
                        return [1 - norm(ψt)^2, norm(ψt)]
                    end
                    function cost(p::Vector)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (exp(p[1]*t)*H1 + exp(p[2]*t)*H2, dagger(exp(p[1]*t)*H1 + exp(p[2]*t)*H2), [], []))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Vector)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (exp(p[1]*t)*H1 + exp(p[2]*t)*H2, dagger(exp(p[1]*t)*H1 + exp(p[2]*t)*H2), [], []))
                        return [1 - norm(ρt)^2, norm(ρt)]
                    end
                    for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                        # out-of-place
                        @test_nowarn derivative(cost, backend, pnum)
                        @test_nowarn second_derivative(cost, backend, pnum)
                        @test_nowarn gradient(cost, backend, pvec)
                        @test_nowarn jacobian(costvec, backend, pvec)
                        @test_nowarn hessian(cost, backend, pvec)
                        # in-place
                        @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn gradient!(cost, bufvec, backend, pvec)
                        @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                        @test_nowarn hessian!(cost, bufmat, backend, pvec)
                    end
                end

                # differentiate jump operators
                @testset "jump operators" begin

                    function cost(p::Real)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (H1, dagger(H1), [exp(p*t)*H2], [dagger(exp(p*t)*H2)]))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Real)
                        _, ψt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (H1, dagger(H1), [exp(p*t)*H2], [dagger(exp(p*t)*H2)]))
                        return [1 - norm(ψt)^2, norm(ψt)]
                    end
                    function cost(p::Vector)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (H1, dagger(H1), [exp(p[1]*t)*H1, exp(p[2]*t)*H2], [dagger(exp(p[1]*t)*H1), dagger(exp(p[2]*t)*H2)]))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Vector)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (H1, dagger(H1), [exp(p[1]*t)*H1, exp(p[2]*t)*H2], [dagger(exp(p[1]*t)*H1), dagger(exp(p[2]*t)*H2)]))
                        return [1 - norm(ρt)^2, norm(ρt)]
                    end
                    for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                        # out-of-place
                        @test_nowarn derivative(cost, backend, pnum)
                        @test_nowarn second_derivative(cost, backend, pnum)
                        @test_nowarn gradient(cost, backend, pvec)
                        @test_nowarn jacobian(costvec, backend, pvec)
                        @test_nowarn hessian(cost, backend, pvec)
                        # in-place
                        @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn gradient!(cost, bufvec, backend, pvec)
                        @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                        @test_nowarn hessian!(cost, bufmat, backend, pvec)
                    end
                end

                # differentiate rates
                @testset "rates" begin

                    function cost(p::Real)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (H1, dagger(H1), [H2], [dagger(H2)], [exp(p*t)]))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Real)
                        _, ψt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (H1, dagger(H1), [H2], [dagger(H2)], [exp(p*t)]))
                        return [1 - norm(ψt)^2, norm(ψt)]
                    end
                    function cost(p::Vector)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (H1, dagger(H1), [H1, H2], [dagger(H1), dagger(H2)], [exp(p[1]*t), exp(p[2]*t)]))
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Vector)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, (t, ρ) -> (H1, dagger(H1), [H1, H2], [dagger(H1), dagger(H2)], [exp(p[1]*t), exp(p[2]*t)]))
                        return [1 - norm(ρt)^2, norm(ρt)]
                    end
                    for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                        # out-of-place
                        @test_nowarn derivative(cost, backend, pnum)
                        @test_nowarn second_derivative(cost, backend, pnum)
                        @test_nowarn gradient(cost, backend, pvec)
                        @test_nowarn jacobian(costvec, backend, pvec)
                        @test_nowarn hessian(cost, backend, pvec)
                        # in-place
                        @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn gradient!(cost, bufvec, backend, pvec)
                        @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                        @test_nowarn hessian!(cost, bufmat, backend, pvec)
                    end
                end
            end

            @testset "TimeDependentOperators" begin

                # differentiate Hamiltonian
                @testset "Hamiltonian" begin

                    function cost(p::Real)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, p*H1t, [p*H1t])
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Real)
                        _, ψt = timeevolution.master_nh_dynamic(tspan, ρ0, p*H1t, [p*H1t])
                        return [1 - norm(ψt)^2, norm(ψt)]
                    end
                    function cost(p::Vector)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, p[1]*H1t, [p[2]*H1t])
                        return 1 - norm(ρt)^2
                    end
                    function costvec(p::Vector)
                        _, ρt = timeevolution.master_nh_dynamic(tspan, ρ0, p[1]*H1t, [p[1]*H1t])
                        return [1 - norm(ρt)^2, norm(ρt)]
                    end
                    for backend in [AutoForwardDiff(), AutoFiniteDiff()]
                        # out-of-place
                        @test_nowarn derivative(cost, backend, pnum)
                        @test_nowarn second_derivative(cost, backend, pnum)
                        @test_nowarn gradient(cost, backend, pvec)
                        @test_nowarn jacobian(costvec, backend, pvec)
                        @test_nowarn hessian(cost, backend, pvec)
                        # in-place
                        @test_nowarn derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn second_derivative!(costvec, bufvec, backend, pnum)
                        @test_nowarn gradient!(cost, bufvec, backend, pvec)
                        @test_nowarn jacobian!(costvec, bufmat, backend, pvec)
                        @test_nowarn hessian!(cost, bufmat, backend, pvec)
                    end
                end
            end
        end
    end
end