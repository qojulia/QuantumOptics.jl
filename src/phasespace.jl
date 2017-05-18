module phasespace

export qfunc, wigner

using ..bases, ..states, ..operators, ..operators_dense, ..fock


"""
    qfunc(a, α)
    qfunc(a, x, y)
    qfunc(a, xvec, yvec)

Husimi Q representation ``⟨α|ρ|α⟩/π`` for the given state or operator `a`. The
function can either be evaluated on one point α or on a grid specified by
the vectors `xvec` and `yvec`. Note that conversion from `x` and `y` to `α` is
done via the relation ``α = \\frac{1}{\\sqrt{2}}(x + i y)``.
"""
function qfunc(rho::Operator, alpha::Number)
    b = basis(rho)
    @assert isa(b, FockBasis)
    _qfunc_operator(rho, convert(Complex128, alpha), Ket(b), Ket(b))
end

function qfunc(rho::Operator, xvec::Vector{Float64}, yvec::Vector{Float64})
    b = basis(rho)
    @assert isa(b, FockBasis)
    Nx = length(xvec)
    Ny = length(yvec)
    tmp1 = Ket(b)
    tmp2 = Ket(b)
    result = Matrix{Complex128}(Nx, Ny)
    @inbounds for j=1:Ny, i=1:Nx
        result[i, j] = _qfunc_operator(rho, complex(xvec[i], yvec[j])/sqrt(2), tmp1, tmp2)
    end
    result
end

function qfunc(psi::Ket, alpha::Number)
    b = basis(psi)
    @assert isa(b, FockBasis)
    _alpha = convert(Complex128, alpha)
    _conj_alpha = conj(_alpha)
    N = length(psi.basis)
    s = psi.data[N]/sqrt(N-1)
    @inbounds for i=1:N-2
        s = (psi.data[N-i] + s*_conj_alpha)/sqrt(N-i-1)
    end
    s = psi.data[1] + s*_conj_alpha
    return abs2(s)*exp(-abs2(_alpha))/pi
end

function qfunc(psi::Ket, xvec::Vector{Float64}, yvec::Vector{Float64})
    b = basis(psi)
    @assert isa(b, FockBasis)
    Nx = length(xvec)
    Ny = length(yvec)
    N = length(b)
    x = similar(psi.data)
    x[N] = psi.data[1]
    n = 1.
    @inbounds for i in 1:N-1
        x[N-i] = psi.data[i+1]/n
        n *= sqrt(i+1)
    end
    result = Matrix{Float64}(Nx, Ny)
    for j=1:Ny, i=1:Nx
        _conj_alpha = complex(xvec[i], -yvec[j])/sqrt(2)
        result[i, j] = _qfunc_ket(x, _conj_alpha)
    end
    return result
end

function qfunc(state::Union{Ket, Operator}, x::Number, y::Number)
    qfunc(state, Complex128(x, y)/sqrt(2))
end

function _qfunc_operator(rho::Operator, alpha::Complex128, tmp1::Ket, tmp2::Ket)
    coherentstate(basis(rho), alpha, tmp1)
    operators.gemv!(complex(1.), rho, tmp1, complex(0.), tmp2)
    a = dot(tmp1.data, tmp2.data)
    return a/pi
end

function _qfunc_ket(x::Vector{Complex128}, conj_alpha::Complex128)
    s = x[1]
    @inbounds for i=2:length(x)
        s = x[i] + s*conj_alpha
    end
    abs2(s)*exp(-abs2(conj_alpha))/pi
end


"""
    wigner(a, α)
    wigner(a, x, y)
    wigner(a, xvec, yvec)

Wigner function for the given state or operator `a`. The
function can either be evaluated on one point α or on a grid specified by
the vectors `xvec` and `yvec`. Note that conversion from `x` and `y` to `α` is
done via the relation ``α = \\frac{1}{\\sqrt{2}}(x + y)``.
"""
function wigner(rho::DenseOperator, x::Number, y::Number)
    b = basis(rho)
    @assert isa(b, FockBasis)
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

function wigner(rho::DenseOperator, xvec::Vector{Float64}, yvec::Vector{Float64})
    b = basis(rho)
    @assert isa(b, FockBasis)
    N = b.N::Int
    _2α = [complex(x, y)*sqrt(2) for x=xvec, y=yvec]
    abs2_2α = abs2(_2α)
    w = zeros(_2α)
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


function _clenshaw_grid(L::Int, ρ::Matrix{Complex128},
                abs2_2α::Matrix{Float64}, _2α::Matrix{Complex128}, w::Matrix{Complex128},
                b0::Matrix{Complex128}, b1::Matrix{Complex128}, b2::Matrix{Complex128}, scale::Int)
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

function _clenshaw(L::Int, abs2_2α::Float64, ρ::Matrix{Complex128})
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

end #module
