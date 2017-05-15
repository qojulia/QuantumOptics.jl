module polynomials

export horner, hermite, laguerre

"""
    horner(coefficients, x)

Evaluate the given polynomial at position x using the Horner scheme.

```math
p(x) = \\sum_{n=0}^N c_n x^n
```
"""
function horner{T<:Number}(coefficients::Vector{T}, x::Number)
    bn = coefficients[end]
    for n=length(coefficients)-1:-1:1
        bn = coefficients[n] + bn*x
    end
    bn
end


module hermite

"""
    a_nk(N)

Calculate the all coefficients for all Hermite polynomials up to order N.

```math
H_n(x) = \\sum_{k=0}^n a_{n,k} x^k
```

Returns a vector of length N+1 where the n-th entry contains all coefficients
for the n-th Hermite polynomial.
"""
function a(N::Int)
    a = Vector{Vector{Int}}(N+1)
    a[1] = [1]
    a[2] = [0,2]
    am = a[2]
    for n=2:N
        an = zeros(Int, n+1)
        a[n+1] = an
        if iseven(n)
            an[1] = -am[2]
        end
        an[n+1] = 2*am[n]
        if iseven(n)
            for k=3:2:n-1
                an[k] = 2*am[k-1] - am[k+1]*k
            end
        else
            for k=2:2:n-1
                an[k] = 2*am[k-1] - am[k+1]*k
            end
        end
        am = an
    end
    a
end

"""
    A_nk(N)

Calculate the all scaled coefficients for all Hermite polynomials up to order N.

The scaled coefficients `A` are connected to the unscaled coefficients `a` by
the relation ``A_{n,k} = \\frac{a_{n,k}}{\\sqrt{2^n n!}}``

Returns a vector of length N+1 where the n-th entry contains all scaled
coefficients for the n-th Hermite polynomial.
"""
function A(N)
    A = Vector{Vector{Float64}}(N+1)
    A[1] = [1.]
    A[2] = [0., sqrt(2)]
    Am = A[2]
    for n=2:N
        An = zeros(Float64, n+1)
        A[n+1] = An
        if iseven(n)
            An[1] = -Am[2]/sqrt(2*n)
        end
        An[n+1] = Am[n]*sqrt(2/n)
        if iseven(n)
            for k=3:2:n-1
                An[k] = Am[k-1]*sqrt(2/n) - Am[k+1]*k/sqrt(2*n)
            end
        else
            for k=2:2:n-1
                An[k] = Am[k-1]*sqrt(2/n) - Am[k+1]*k/sqrt(2*n)
            end
        end
        Am = An
    end
    A
end

end # module hermite


module laguerre

end # module laguerre

end # module polynomials