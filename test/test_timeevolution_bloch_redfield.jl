@testitem "bloch-redfield" begin
using QuantumOptics

Δ = 0.2 * 2*π
ϵ0 = 1.0 * 2*π
γ1 = 0.5

b = SpinBasis(1//2)
sx = sigmax(b)
sz = sigmaz(b)

H = -Δ/2.0 * sx - ϵ0/2.0 * sz

function ohmic_spectrum(ω)
    ω == 0.0 ? (return γ1) : (return γ1/2 * (ω / (2*π)) * (ω > 0.0))
end

R, ekets = timeevolution.bloch_redfield_tensor(H, [[sx, ohmic_spectrum]])

known_result =  [0.0+0.0im        0.0+0.0im            0.0+0.0im       0.245145+0.0im
 0.0+0.0im  -0.161034-6.40762im        0.0+0.0im            0.0+0.0im
 0.0+0.0im        0.0+0.0im      -0.161034+6.40762im        0.0+0.0im
 0.0+0.0im        0.0+0.0im            0.0+0.0im      -0.245145+0.0im]
@test isapprox(dense(R).data, known_result, atol=1e-5)

psi0 = spindown(b)
tout, ρt = timeevolution.master_bloch_redfield([0.0:0.1:2.0;], psi0, R, H)
rho_end = [0.38206-0.0im 0.0466443+0.0175017im
            0.0466443-0.0175017im 0.61794+0.0im]
@test isapprox(ρt[end].data, rho_end, atol=1e-5)
@test ρt[end] != ρt[end-1]
@test isa(ρt, Vector{<:DenseOpType})

# Test fout
fout(t,rho) = copy(rho)
tout, ρt2 = timeevolution.master_bloch_redfield([0.0:0.1:2.0;], psi0, R, H; fout=fout)
@test all(ρt .== ρt2)

fout2(t,rho) = expect(sz,rho)
tout, z = timeevolution.master_bloch_redfield([0.0:0.1:2.0;], psi0, R, H; fout=fout2)
@test length(z) == length(ρt)
@test isa(first(z), ComplexF64)

end # testset
