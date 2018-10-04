using Test
using QuantumOptics
using LinearAlgebra

@testset "spin" begin

D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))

# Test creation
@test_throws AssertionError SpinBasis(1//3)
@test_throws AssertionError SpinBasis(-1//2)
@test_throws AssertionError SpinBasis(0)


for spinnumber=[1//2, 1, 3//2, 4//2]
    spinbasis = SpinBasis(spinnumber)
    I = operators.identityoperator(spinbasis)
    Zero = Operator(spinbasis)
    sx = sigmax(spinbasis)
    sy = sigmay(spinbasis)
    sz = sigmaz(spinbasis)
    sp = sigmap(spinbasis)
    sm = sigmam(spinbasis)


    # Test traces
    @test 0 == tr(sx)
    @test 0 == tr(sy)
    @test 0 == tr(sz)


    # Test kommutation relations
    kommutator(x, y) = x*y - y*x

    @test 1e-12 > D(kommutator(sx, sx), Zero)
    @test 1e-12 > D(kommutator(sx, sy), 2im*sz)
    @test 1e-12 > D(kommutator(sx, sz), -2im*sy)
    @test 1e-12 > D(kommutator(sy, sx), -2im*sz)
    @test 1e-12 > D(kommutator(sy, sy), Zero)
    @test 1e-12 > D(kommutator(sy, sz), 2im*sx)
    @test 1e-12 > D(kommutator(sz, sx), 2im*sy)
    @test 1e-12 > D(kommutator(sz, sy), -2im*sx)
    @test 1e-12 > D(kommutator(sz, sz), Zero)


    # Test creation and anihilation operators
    @test 0 == D(sp, 0.5*(sx + 1im*sy))
    @test 0 == D(sm, 0.5*(sx - 1im*sy))
    @test 0 == D(sx, (sp + sm))
    @test 0 == D(sy, -1im*(sp - sm))


    # Test commutation relations with creation and anihilation operators
    @test 1e-12 > D(kommutator(sp, sm), sz)
    @test 1e-12 > D(kommutator(sz, sp), 2*sp)
    @test 1e-12 > D(kommutator(sz, sm), -2*sm)


    # Test v x (v x u) relation: [sa, [sa, sb]] = 4*(1-delta_{ab})*sb
    @test 1e-12 > D(kommutator(sx, kommutator(sx, sx)), Zero)
    @test 1e-12 > D(kommutator(sx, kommutator(sx, sy)), 4*sy)
    @test 1e-12 > D(kommutator(sx, kommutator(sx, sz)), 4*sz)
    @test 1e-12 > D(kommutator(sy, kommutator(sy, sx)), 4*sx)
    @test 1e-12 > D(kommutator(sy, kommutator(sy, sy)), Zero)
    @test 1e-12 > D(kommutator(sy, kommutator(sy, sz)), 4*sz)
    @test 1e-12 > D(kommutator(sz, kommutator(sz, sx)), 4*sx)
    @test 1e-12 > D(kommutator(sz, kommutator(sz, sy)), 4*sy)
    @test 1e-12 > D(kommutator(sz, kommutator(sz, sz)), Zero)


    # Test spinup and spindown states
    @test 1 ≈ norm(spinup(spinbasis))
    @test 1 ≈ norm(spindown(spinbasis))
    @test 0 ≈ norm(sp*spinup(spinbasis))
    @test 0 ≈ norm(sm*spindown(spinbasis))
end


# Test special relations for spin 1/2

spinbasis = SpinBasis(1//2)
I = identityoperator(spinbasis)
Zero = Operator(spinbasis)
sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)


# Test antikommutator
antikommutator(x, y) = x*y + y*x

@test 0 ≈ D(antikommutator(sx, sx), 2*I)
@test 0 ≈ D(antikommutator(sx, sy), Zero)
@test 0 ≈ D(antikommutator(sx, sz), Zero)
@test 0 ≈ D(antikommutator(sy, sx), Zero)
@test 0 ≈ D(antikommutator(sy, sy), 2*I)
@test 0 ≈ D(antikommutator(sy, sz), Zero)
@test 0 ≈ D(antikommutator(sz, sx), Zero)
@test 0 ≈ D(antikommutator(sz, sy), Zero)
@test 0 ≈ D(antikommutator(sz, sz), 2*I)


# Test if involutory for spin 1/2
@test 0 ≈ D(sx*sx, I)
@test 0 ≈ D(sy*sy, I)
@test 0 ≈ D(sz*sz, I)
@test 0 ≈ D(-1im*sx*sy*sz, I)


# Test consistency of spin up and down with sigmap and sigmam
@test 1e-11 > norm(sm*spinup(spinbasis) - spindown(spinbasis))
@test 1e-11 > norm(sp*spindown(spinbasis) - spinup(spinbasis))

end # testset
