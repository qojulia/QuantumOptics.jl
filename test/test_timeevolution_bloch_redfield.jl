using QuantumOptics
using Test


@testset "bloch-redfield" begin

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


end # testset
