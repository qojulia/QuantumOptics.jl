module phasespace

export qfunc, wigner, coherentspinstate, qfuncsu2, wignersu2, ylm

using ..bases, ..states, ..operators, ..operators_dense, ..fock, ..spin
using LinearAlgebra

import WignerSymbols: clebschgordan

"""
    qfunc(a, α)
    qfunc(a, x, y)
    qfunc(a, xvec, yvec)

Husimi Q representation ``⟨α|ρ|α⟩/π`` for the given state or operator `a`. The
function can either be evaluated on one point α or on a grid specified by
the vectors `xvec` and `yvec`. Note that conversion from `x` and `y` to `α` is
done via the relation ``α = \\frac{1}{\\sqrt{2}}(x + i y)``.
"""
function qfunc(rho::AbstractOperator{B,B}, alpha::Number) where B<:FockBasis
    b = basis(rho)
    _qfunc_operator(rho, convert(ComplexF64, alpha), Ket(b), Ket(b))
end

function qfunc(rho::AbstractOperator{B,B}, xvec::Vector{Float64}, yvec::Vector{Float64}) where B<:FockBasis
    b = basis(rho)
    Nx = length(xvec)
    Ny = length(yvec)
    tmp1 = Ket(b)
    tmp2 = Ket(b)
    result = Matrix{ComplexF64}(undef, Nx, Ny)
    @inbounds for j=1:Ny, i=1:Nx
        result[i, j] = _qfunc_operator(rho, complex(xvec[i], yvec[j])/sqrt(2), tmp1, tmp2)
    end
    result
end

function qfunc(psi::Ket{B}, alpha::Number) where B<:FockBasis
    b = basis(psi)
    _alpha = convert(ComplexF64, alpha)
    _conj_alpha = conj(_alpha)
    N = length(psi.basis)
    s = psi.data[N]/sqrt(N-1)
    @inbounds for i=1:N-2
        s = (psi.data[N-i] + s*_conj_alpha)/sqrt(N-i-1)
    end
    s = psi.data[1] + s*_conj_alpha
    return abs2(s)*exp(-abs2(_alpha))/pi
end

function qfunc(psi::Ket{B}, xvec::Vector{Float64}, yvec::Vector{Float64}) where B<:FockBasis
    b = basis(psi)
    Nx = length(xvec)
    Ny = length(yvec)
    points = Nx*Ny
    N = length(b)::Int
    _conj_alpha = [complex(x, -y)/sqrt(2) for x=xvec, y=yvec]
    q = fill(psi.data[N]/sqrt(N-1), size(_conj_alpha))
    @inbounds for n=1:N-2
        f0_ = 1/sqrt(N-n-1)
        x = psi.data[N-n]
        for i=1:points
            q[i] = (x + q[i]*_conj_alpha[i])*f0_
        end
    end
    result = similar(q, Float64)
    x = psi.data[1]
    @inbounds for i=1:points
        result[i] = abs2(x + q[i]*_conj_alpha[i])*exp(-abs2(_conj_alpha[i]))/pi
    end
    result
end

function qfunc(state::Union{Ket{B}, AbstractOperator{B,B}}, x::Number, y::Number) where B<:FockBasis
    qfunc(state, ComplexF64(x, y)/sqrt(2))
end

function _qfunc_operator(rho::AbstractOperator{B,B}, alpha::ComplexF64, tmp1::Ket, tmp2::Ket) where B<:FockBasis
    coherentstate(basis(rho), alpha, tmp1)
    operators.gemv!(complex(1.), rho, tmp1, complex(0.), tmp2)
    a = dot(tmp1.data, tmp2.data)
    return a/pi
end


"""
    wigner(a, α)
    wigner(a, x, y)
    wigner(a, xvec, yvec)

Wigner function for the given state or operator `a`. The
function can either be evaluated on one point α or on a grid specified by
the vectors `xvec` and `yvec`. Note that conversion from `x` and `y` to `α` is
done via the relation ``α = \\frac{1}{\\sqrt{2}}(x + i y)``.
"""
function wigner(rho::DenseOperator{B,B}, x::Number, y::Number) where B<:FockBasis
    b = basis(rho)
    N = b.N::Int
    _2α = complex(convert(Float64, x), convert(Float64, y))*sqrt(2)
    abs2_2α = abs2(_2α)
    w = complex(0.)
    coefficient = complex(0.)
    @inbounds for L=N:-1:1
        coefficient = 2*_clenshaw(L, abs2_2α, rho.data)
        w = coefficient + w*_2α/sqrt(L+1)
    end
    coefficient = _clenshaw(0, abs2_2α, rho.data)
    w = coefficient + w*_2α
    exp(-abs2_2α/2)/pi*real(w)
end

function wigner(rho::DenseOperator{B,B}, xvec::Vector{Float64}, yvec::Vector{Float64}) where B<:FockBasis
    b = basis(rho)
    N = b.N::Int
    _2α = [complex(x, y)*sqrt(2) for x=xvec, y=yvec]
    abs2_2α = abs2.(_2α)
    w = zero(_2α)
    b0 = similar(_2α)
    b1 = similar(_2α)
    b2 = similar(_2α)
    @inbounds for L=N:-1:1
        _clenshaw_grid(L, rho.data, abs2_2α, _2α, w, b0, b1, b2, 2)
    end
    _clenshaw_grid(0, rho.data, abs2_2α, _2α, w, b0, b1, b2, 1)
    @inbounds for i=eachindex(w)
        abs2_2α[i] = exp(-abs2_2α[i]/2)/pi.*real(w[i])
    end
    abs2_2α
end

wigner(psi::Ket, x, y) = wigner(dm(psi), x, y)
wigner(state, alpha::Number) = wigner(state, real(alpha)*sqrt(2), imag(alpha)*sqrt(2))


function _clenshaw_grid(L::Int, ρ::Matrix{ComplexF64},
                abs2_2α::Matrix{Float64}, _2α::Matrix{ComplexF64}, w::Matrix{ComplexF64},
                b0::Matrix{ComplexF64}, b1::Matrix{ComplexF64}, b2::Matrix{ComplexF64}, scale::Int)
    n = size(ρ, 1)-L-1
    points = length(w)
    if n==0
        f = scale*ρ[1, L+1]
        @inbounds for i=1:points
            w[i] = f + w[i]*_2α[i]/sqrt(L+1)
        end
    elseif n==1
        f1 = 1/sqrt(L+1)
        @inbounds for i=1:points
            w[i] = scale*(ρ[1, L+1] - ρ[2, L+2]*(L+1-abs2_2α[i])*f1) + w[i]*_2α[i]*f1
        end
    else
        f0 = sqrt(float((n+L-1)*(n-1)))
        f1 = sqrt(float((n+L)*n))
        f0_ = 1/f0
        f1_ = 1/f1
        fill!(b1, ρ[n+1, L+n+1])
        @inbounds for i=1:points
            b0[i] = ρ[n, L+n] - (2*n-1+L-abs2_2α[i])*f1_*b1[i]
        end
        @inbounds for k=n-2:-1:1
            b1, b2, b0 = b0, b1, b2
            x = ρ[k+1, L+k+1]
            a1 = -(2*k+1+L)
            a2 = -f0*f1_
            @inbounds for i=1:points
                b0[i] = x + (a1+abs2_2α[i])*f0_*b1[i] + a2*b2[i]
            end
            f1 , f1_ = f0, f0_
            f0 = sqrt((k+L)*k)
            f0_ = 1/f0
        end
        @inbounds for i=1:points
            w[i] = scale*(ρ[1, L+1] - (L+1-abs2_2α[i])*f0_*b0[i] - f0*f1_*b1[i]) + w[i]*_2α[i]*f0_
        end
    end
end

function _clenshaw(L::Int, abs2_2α::Float64, ρ::Matrix{ComplexF64})
    n = size(ρ, 1)-L-1
    if n==0
        return ρ[1, L+1]
    elseif n==1
        ϕ1 = -(L+1-abs2_2α)/sqrt(L+1)
        return ρ[1, L+1] + ρ[2, L+2]*ϕ1
    else
        f0 = sqrt(float((n+L-1)*(n-1)))
        f1 = sqrt(float((n+L)*n))
        f0_ = 1/f0
        f1_ = 1/f1
        b2 = complex(0.)
        b1 = ρ[n+1, L+n+1]
        b0 = ρ[n, L+n] - (2*n-1+L-abs2_2α)*f1_*b1
        @inbounds for k=n-2:-1:1
            b1, b2 = b0, b1
            b0 = ρ[k+1, L+k+1] - (2*k+1+L-abs2_2α)*f0_*b1 - f0*f1_*b2
            f1, f1_ = f0, f0_
            f0 = sqrt((k+L)*k)
            f0_ = 1/f0
        end
        return ρ[1, L+1] - (L+1-abs2_2α)*f0_*b0 - f0*f1_*b1
    end
end

"""
    coherentspinstate(b::SpinBasis, θ::Real, ϕ::Real)

A coherent spin state |θ,ϕ⟩ is analogous to the coherent state of the linear harmonic
oscillator. Coherent spin states represent a collection of identical two-level
systems and can be described by two angles θ and ϕ (although this
parametrization is not unique), similarly to a qubit on the
Bloch sphere.
"""
function coherentspinstate(b::SpinBasis, theta::Real, phi::Real,
    result = Ket(b, Vector{ComplexF64}(undef, length(b))))
    data = result.data
    N = BigInt(length(b)-1)
    sinth = sin(0.5theta)
    costh = cos(0.5theta)
    expphi = exp(0.5im*phi)
    expphi_con = conj(expphi)
    @inbounds for n=0:N
        data[n+1] = sqrt(binomial(N, n)) * (sinth*expphi)^n * (costh*expphi_con)^(N-n)
    end
    return result
end

"""
    qfuncsu2(ket,Ntheta;Nphi=2Ntheta)
    qfuncsu2(rho,Ntheta;Nphi=2Ntheta)

Husimi Q SU(2) representation ``⟨θ,ϕ|ρ|θ,ϕ⟩/π`` for the given state.

The function calculates the SU(2) Husimi representation of a state on the
generalised bloch sphere (0 < θ < π and 0 < ϕ < 2 π) with a given
resolution `(Ntheta, Nphi)`.

    qfuncsu2(rho,θ,ϕ)
    qfuncsu2(ket,θ,ϕ)

This version calculates the Husimi Q SU(2) function at a position given by θ and ϕ.
"""
function qfuncsu2(psi::Ket{B}, Ntheta::Int; Nphi::Int=2Ntheta) where B<:SpinBasis
    b = psi.basis
    psi_bra_data = psi.data'
    lb = float(b.spinnumber)
    result = Array{Float64}(undef, Ntheta,Nphi)
    @inbounds  for i = 0:Ntheta-1, j = 0:Nphi-1
        result[i+1,j+1] = (2*lb+1)/(4pi)*abs2(psi_bra_data*coherentspinstate(b,pi-i*pi/(Ntheta-1),j*2pi/(Nphi-1)-pi).data)
    end
    return result
end

function qfuncsu2(rho::DenseOperator{B,B}, Ntheta::Int; Nphi::Int=2Ntheta) where B<:SpinBasis
    b = basis(rho)
    lb = float(b.spinnumber)
    result = Array{Float64}(undef, Ntheta,Nphi)
    @inbounds  for i = 0:Ntheta-1, j = 0:Nphi-1
        c = coherentspinstate(b,pi-i*1pi/(Ntheta-1),j*2pi/(Nphi-1)-pi)
        result[i+1,j+1] = abs((2*lb+1)/(4pi)*c.data'*rho.data*c.data)
    end
    return result
end

function qfuncsu2(psi::Ket{B}, theta::Real, phi::Real) where B<:SpinBasis
    b = basis(psi)
    psi_bra_data = psi.data'
    lb = float(b.spinnumber)
    result = (2*lb+1)/(4pi)*abs2(psi_bra_data*coherentspinstate(b,theta,phi).data)
    return result
end

function qfuncsu2(rho::DenseOperator{B,B}, theta::Real, phi::Real) where B<:SpinBasis
    b = basis(rho)
    lb = float(b.spinnumber)
    c = coherentspinstate(b,theta,phi)
    result = abs((2*lb+1)/(4pi)*c.data'*rho.data*c.data)
    return result
end

"""
    wignersu2(ket,Ntheta;Nphi=2Ntheta)
    wignersu2(rho,Ntheta;Nphi=2Ntheta)

Wigner SU(2) representation for the given state with a resolution `(Ntheta, Nphi)`.

The function calculates the SU(2) Wigner representation of a state on the
generalised bloch sphere (0 < θ < π and 0 < ϕ < 2 π) with a given resolution by
decomposing the state into the basis of spherical harmonics.

    wignersu2(rho,θ,ϕ)
    wignersu2(ket,θ,ϕ)

This version calculates the Wigner SU(2) function at a position given by θ and ϕ
"""
function wignersu2(rho::DenseOperator{B,B}, theta::Real, phi::Real) where B<:SpinBasis

    N = length(basis(rho))-1

    ### Tensor generation ###
    BandT = Array{Vector{Float64}}(undef, N,N+1)
    BandT[1,1] = collect(range(-N/2, stop=N/2, length=N+1))
    BandT[1,2] = -collect(sqrt.(range(1, stop=N, length=N)).*sqrt.(range((N)/2, stop=1/2, length=N)))
    BandT[2,1] = clebschgordan(1,0,1,0,2,0)*BandT[1,1].*BandT[1,1] -
        clebschgordan(1,-1,1,1,2,0)*[zeros(N+1-length(BandT[1,2])); BandT[1,2].*BandT[1,2]] -
        clebschgordan(1,1,1,-1,2,0)*[BandT[1,2].*BandT[1,2]; zeros(N+1-length(BandT[1,2]))]
    BandT[2,2] = clebschgordan(1,0,1,1,2,1)BandT[1,1][1:N].*BandT[1,2]+
        clebschgordan(1,1,1,0,2,1)*BandT[1,2][1:N].*BandT[1,1][2:end]
    BandT[2,3] = BandT[1,2][1:N+1-(2)].*BandT[1,2][2:end]

    @inbounds for S=2:N-1
        BandT[S+1,1] = clebschgordan(1,0,S,0,S+1,0)*BandT[1,1].*BandT[S,1] -
            [zeros(N+1-length(BandT[1,2])); clebschgordan(1,-1,S,1,S+1,0)*BandT[1,2].*BandT[S,2]] -
            clebschgordan(1,1,S,-1,S+1,0)*[BandT[1,2].*BandT[S,2]; zeros(N+1-length(BandT[1,2]))]
        BandT[S+1,S+1] = clebschgordan(1,0,S,S,S+1,S)BandT[1,1][1:N+1-S].*BandT[S,S+1]+
            clebschgordan(1,1,S,S-1,S+1,S)*BandT[1,2][1:N+1-S].*BandT[S,S][2:end]
        BandT[S+1,S+2] = BandT[1,2][1:N+1-(S+1)].*BandT[S,S+1][2:end]
        @inbounds for M=1:S-1
            BandT[S+1,M+1] = clebschgordan(1, 0, S, M, S+1,M)*BandT[1,1][1:N+1-M].*BandT[S,M+1] +
                clebschgordan(1,1,S,M-1,S+1,M)*BandT[1,2][1:N+1-M].*BandT[S,M][2:end] -
                clebschgordan(1,-1,S,M+1,S+1,M)*[zeros(1); BandT[1,2][1:N-M].*BandT[S,M+2][1:N-M]]
        end

    end

    NormT = zeros(N)
    @inbounds for S = 1:N
        NormT[S] = sum(BandT[S,1].^2)
    end

    @inbounds for S = 1:N, M = 0:S
        BandT[S, M + 1] =  BandT[S, M + 1]/sqrt(NormT[S])
    end

    ### State decomposition ###
    c = rho.data
    EVT = Array{ComplexF64}(undef, N,N+1)
    @inbounds for S = 1:N, M = 0:S
        EVT[S,M+1] = conj(sum(BandT[S,M+1].*diag(c,M)))
    end

    wignermap = _wignersu2int(N,theta,phi, EVT)
    return wignermap*sqrt((N+1)/(4pi))
end

function wignersu2(rho::DenseOperator{B,B}, Ntheta::Int; Nphi::Int=2Ntheta) where B<:SpinBasis

    N = length(basis(rho))-1

    ### Tensor generation ###
    BandT = Array{Vector{Float64}}(undef, N,N+1)
    BandT[1,1] = collect(range(-N/2, stop=N/2, length=N+1))
    BandT[1,2] = -collect(sqrt.(range(1, stop=N, length=N)).*sqrt.(range((N)/2, stop=1/2, length=N)))
    BandT[2,1] = clebschgordan(1,0,1,0,2,0)*BandT[1,1].*BandT[1,1] -
        clebschgordan(1,-1,1,1,2,0)*[zeros(N+1-length(BandT[1,2])); BandT[1,2].*BandT[1,2]] -
        clebschgordan(1,1,1,-1,2,0)*[BandT[1,2].*BandT[1,2]; zeros(N+1-length(BandT[1,2]))]
    BandT[2,2] = clebschgordan(1,0,1,1,2,1)BandT[1,1][1:N].*BandT[1,2]+
        clebschgordan(1,1,1,0,2,1)*BandT[1,2][1:N].*BandT[1,1][2:end]
    BandT[2,3] = BandT[1,2][1:N+1-(2)].*BandT[1,2][2:end]

    @inbounds for S=2:N-1
        BandT[S+1,1] = clebschgordan(1,0,S,0,S+1,0)*BandT[1,1].*BandT[S,1] -
            [zeros(N+1-length(BandT[1,2])); clebschgordan(1,-1,S,1,S+1,0)*BandT[1,2].*BandT[S,2]] -
            clebschgordan(1,1,S,-1,S+1,0)*[BandT[1,2].*BandT[S,2]; zeros(N+1-length(BandT[1,2]))]
        BandT[S+1,S+1] = clebschgordan(1,0,S,S,S+1,S)BandT[1,1][1:N+1-S].*BandT[S,S+1]+
            clebschgordan(1,1,S,S-1,S+1,S)*BandT[1,2][1:N+1-S].*BandT[S,S][2:end]
        BandT[S+1,S+2] = BandT[1,2][1:N+1-(S+1)].*BandT[S,S+1][2:end]
        @inbounds for M=1:S-1
            BandT[S+1,M+1] = clebschgordan(1, 0, S, M, S+1,M)*BandT[1,1][1:N+1-M].*BandT[S,M+1] +
                clebschgordan(1,1,S,M-1,S+1,M)*BandT[1,2][1:N+1-M].*BandT[S,M][2:end] -
                clebschgordan(1,-1,S,M+1,S+1,M)*[0.0im; BandT[1,2][1:N-M].*BandT[S,M+2][1:N-M]]
        end
    end

    NormT = zeros(N)
    @inbounds for S = 1:N
        NormT[S] = sum(BandT[S,1].^2)
    end
    @inbounds for S = 1:N, M = 0:S
        BandT[S, M + 1] = BandT[S, M + 1]/sqrt(NormT[S])
    end

    ### State decomposition ###
    c = rho.data
    EVT = Array{ComplexF64}(undef, N,N+1)
    @inbounds for S = 1:N, M = 0:S
        EVT[S,M+1] = conj(sum(BandT[S,M+1].*diag(c,M)))
    end

    wignermap = Array{Float64}(undef, Ntheta,Nphi)
    @inbounds for i = 1:Ntheta, j = 1:Nphi
        wignermap[i,j] = _wignersu2int(N,i*1pi/(Ntheta-1)-1pi/(Ntheta-1),j*2pi/(Nphi-1)-2pi/(Nphi-1)-pi, EVT)
    end
    return wignermap*sqrt((N+1)/(4pi))
end

function _wignersu2int(N::Integer, theta::Real, phi::Real, EVT::Array{ComplexF64, 2})
    UberBand = sqrt(1/(1+N))*ylm(0,0,theta,phi)
    @inbounds for S = 1:N
        @inbounds for M = 1:S
            UberBand += 2*real(EVT[S,M+1]*conj(ylm(S,M,theta,phi)))
        end
        UberBand += EVT[S,1]*ylm(S,0,theta,phi)
    end
    UberBand
end
wignersu2(psi::Ket, args...) = wignersu2(dm(psi), args...)

"""
    ylm(l::Integer,m::Integer,theta::Real,phi::Real)

Spherical harmonics Y(l,m)(θ,ϕ) where l ∈ N,  m = -l,-l+1,...,l-1,l, θ ∈ [0,π],
and ϕ ∈ [0,2π).

This function calculates the value of Y(l,m) spherical harmonic at position θ and ϕ.
"""
function ylm(l::Integer,m::Integer,theta::Real,phi::Real)
    phi_ = mod(phi,2pi)
    theta_ = mod(theta,2pi)
    phase = exp(1.0im*m*phi_)
    if theta_ ≈ 0
        if m == 0
            return @. phase*sqrt((2*l+1)/pi)/2
        else
            return 0
        end
    elseif theta_ ≈ pi
        if m == 0
            return @. phase*(-1)^l*sqrt((2*l+1)/pi)/2
        else
            return 0
        end
    else
        if l == 0
            return 1.0/sqrt(4pi)
        else
            m_ = abs(m)
            norm = _calc_ylm_norm(l, m_)
            sign = m > 0 ? (-1)^m_ : 1
            arg = cos(theta_)
            p_ll = 1.0
            @inbounds for fact = 1.0:l
                p_ll *= @. 1.0/((2*fact))*sqrt(1-arg^2)
            end

            if m_ == l
                return @. p_ll/norm*phase*sign
            elseif l-m_ == 1
                p_llp1 = @. 2*l*arg/sqrt(1-arg^2)*p_ll
                return @. p_llp1/norm*phase*sign
            else
                p_llp1 = @. 2*l*arg/sqrt(1-arg^2)*p_ll
                @inbounds for mr = -l:-m_-2
                    p_llp2 = @. -2*(mr+1)*arg/sqrt(1-arg^2)*p_llp1-(l-mr)*(l+mr+1)*p_ll
                    p_ll = p_llp1
                    p_llp1 = p_llp2
                end
                return @. p_llp1/norm*phase*sign
            end
        end
    end
end

function _calc_ylm_norm(l::Int, m_::Int)
    # TODO: Clean up Int types
    if 0 < l+m_ <= 33
        norm = @. Float64(sqrt(4pi)/sqrt(2*l+1)*sqrt(factorial(Int128(l-m_)))/sqrt(factorial(Int128(l+m_))))
    elseif 0 < l-m_ <= 33 && l+m_ > 33
        norm = @. Float64(sqrt(4pi)/sqrt(2*l+1)*sqrt(factorial(Int128(l-m_)))/sqrt(factorial(BigInt(l+m_))))
    else
        norm = @. Float64(sqrt(4pi)/sqrt(2*l+1)*sqrt(factorial(BigInt(l-m_)))/sqrt(factorial(BigInt(l+m_))))
    end
    norm
end

end #module
