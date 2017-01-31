using Base.Test
using QuantumOptics

T = [0.:0.1:0.5;]

# Define Spin 1/2 operators
spinbasis = SpinBasis(1//2)
I = full(identityoperator(spinbasis))
sigmax = full(spin.sigmax(spinbasis))
sigmay = full(spin.sigmay(spinbasis))
sigmaz = full(spin.sigmaz(spinbasis))
sigmap = full(spin.sigmap(spinbasis))
sigmam = full(spin.sigmam(spinbasis))
I_spin = identityoperator(spinbasis)

N = 2
b = tensor([spinbasis for i=1:N]...)

# psi0 = 1./sqrt(2)*(spinup(spinbasis) + spindown(spinbasis))
psi0 = normalize(spinup(spinbasis) + 0.5*spindown(spinbasis))
rho0 = cumulantexpansion.ProductDensityOperator([psi0 for i=1:N]...)

Ω = [1. 2. 3.;
     2. 1. 4.;
     3. 4. 1.]
# Ω = zeros(Float64, N, N)
γ = 1.
δ = 0.2

H = LazyTensor[LazyTensor(b, i, sigmaz, 0.5*δ) for i=1:N]
for i=1:N, j=1:N
    if i==j
        continue
    end
    h = LazyTensor(b, [i, j], [sigmap, sigmam], Ω[i, j])
    push!(H, h)
end
H = LazySum(H...)
J = LazyTensor[LazyTensor(b, i, sigmam, γ) for i=1:N]
# J = LazyTensor[]

tout, rho_t = cumulantexpansion.master(T, rho0, H, J)
tout, rho_t_full = timeevolution.master(T, full(rho0), full(H), map(full, J))

type ProductState
    N::Int
    data::Vector{Float64}
end

ProductState(N::Int) = ProductState(N, zeros(Float64, 3*N))

function ProductState(rho::DenseOperator)
    state = ProductState(N)
    sx, sy, sz = splitstate(state)
    f(ind, op) = real(expect(embed(b, ind, op), rho))
    for k=1:N
        sx[k] = f(k, sigmax)
        sy[k] = f(k, sigmay)
        sz[k] = f(k, sigmaz)
    end
    combinestate(sx, sy, sz, state.data)
    return state
end

splitstate(N::Int, data::Vector{Float64}) = vec(data[0*N+1:1*N]), vec(data[1*N+1:2*N]), vec(data[2*N+1:3*N])
splitstate(state::ProductState) = splitstate(state.N, state.data)
function combinestate(sx::Vector{Float64}, sy::Vector{Float64}, sz::Vector{Float64}, state::Vector{Float64})
    state[0*N+1:1*N] = sx
    state[1*N+1:2*N] = sy
    state[2*N+1:3*N] = sz
    state
end

function f(t, y::Vector{Float64}, dy::Vector{Float64})
    sx, sy, sz = splitstate(N, y)
    dsx, dsy, dsz = splitstate(N, dy)
    @inbounds for k=1:N
        dsx[k] = -δ*sy[k] - 0.5*γ*sx[k]
        dsy[k] = δ*sx[k] - 0.5*γ*sy[k]
        dsz[k] = -γ*(1+sz[k])
        for j=1:N
            if j==k
                continue
            end
            dsx[k] += Ω[k,j]*sy[j]*sz[k]
            dsy[k] += -Ω[k,j]*sx[j]*sz[k]
            dsz[k] += Ω[k,j]*(sx[j]*sy[k] - sy[j]*sx[k])
        end
    end
    combinestate(dsx, dsy, dsz, dy)
end

t_out = Float64[]
state_out = ProductState[]

function fout_(t, y::Vector{Float64})
    push!(t_out, t)
    push!(state_out, ProductState(N, deepcopy(y)))
end

function densityoperator(sx::Real, sy::Real, sz::Real)
    return 0.5*(I + sx*sigmax + sy*sigmay + sz*sigmaz)
end
function densityoperator(state::ProductState)
    sx, sy, sz = splitstate(state)
    rho = densityoperator(sx[1], sy[1], sz[1])
    for i=2:state.N
        rho = tensor(rho, densityoperator(sx[i], sy[i], sz[i]))
    end
    return rho
end

state0 = ProductState(full(rho0))
QuantumOptics.ode_dopri.ode(f, T, state0.data, fout_)

@test tracedistance(densityoperator(state_out[end]), full(rho_t[end])) < 1e-6
# println(tracedistance(densityoperator(state_out[end]), full(rho_t_full[end])))
@test tracedistance(full(rho_t[end]), full(rho_t[1])) > 0.1
