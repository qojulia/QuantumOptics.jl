
function substep{T}(a::Matrix{T}, beta::T, b::Matrix{T}, result::Matrix{T})
    for j=1:size(a,2)
        @simd for i=1:size(b,1)
            @inbounds result[i,j] = a[i,j] + beta*b[i,j]
        end
    end
end

function scale{T}(alpha::T, a::Matrix{T}, result::Matrix{T})
    for j=1:size(a,2)
        for i=1:size(a,1)
            @inbounds result[i,j] = alpha*a[i,j]
        end
    end
end


function oderk_step{T}(F::Function, t::T, h::T, x::Matrix{T}, a::Matrix{T},
                bs::Vector{T}, bp::Vector{T}, c::Vector{T},
                xs::Matrix{T}, xp::Matrix{T}, dx::Matrix{T}, tmp::Matrix{T},
                k::Vector{Matrix{T}})
    substep(x, h*bs[1], k[1], xs) #xs = x + h*bs[1]*k[1]
    substep(x, h*bp[1], k[1], xp) #xp = x + h*bp[1]*k[1]
    for j = 2:length(c)
        scale(a[j,1], k[1], dx) #dx = a[j,1]*k[1]
        for i = 2:j-1
            substep(dx, a[j,i], k[i], dx) #dx += a[j,i]*k[i]
        end
        #k[j] = F(t + h*c[j], x + h*dx)
        substep(x, h, dx, tmp)
        F(t + h*c[j], tmp, k[j])
        # compute the (p-1)th order estimate
        substep(xs, h*bs[j], k[j], xs) #xs = xs + h*bs[j]*k[j]
        # compute the pth order estimate
        substep(xp, h*bp[j], k[j], xp) #xp = xp + h*bp[j]*k[j]
    end
    # Estimate error
    substep(xs, complex(-1.), xp, tmp) #gamma1 = xs - xp
    return norm(reshape(tmp, length(tmp)), Inf)
end


const dp_coefficients = (5,
                         [    0           0          0         0         0        0
                              1/5         0          0         0         0        0
                              3/40        9/40       0         0         0        0
                             44/45      -56/15      32/9       0         0        0
                          19372/6561 -25360/2187 64448/6561 -212/729     0        0
                           9017/3168   -355/33   46732/5247   49/176 -5103/18656  0
                             35/384       0        500/1113  125/192 -2187/6784  11/84],
                         # 4th order b-coefficients
                         [5179/57600 0 7571/16695 393/640 -92097/339200 187/2100 1/40],
                         # 5th order b-coefficients
                         [35/384 0 500/1113 125/192 -2187/6784 11/84 0],
                         )
const c_compl = vec(complex(sum(dp_coefficients[2], 2)))   # consistency condition
const a_compl = complex(dp_coefficients[2])
const bs_compl = vec(complex(dp_coefficients[3]))
const bp_compl = vec(complex(dp_coefficients[4]))

const a2 = Float64[1/5]
const a3 = Float64[3/40, 9/40]
const a4 = Float64[44/45, -56/15, 32/9]
const a5 = Float64[19372/6561, -25360/2187, 64448/6561, -212/729]
const a6 = Float64[9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
const a7 = Float64[35/384, 0., 500/1113, 125/192, -2187/6784, 11/84]
const a = Vector{Float64}[Float64[],a2,a3,a4,a5,a6,a7}
const bp = a7
const bs = Float64[5179/57600, 0., 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
const c = Float64[0., 1/5, 3/10, 4/5, 8/9, 1., 1.]

function substep2{T}(x::Vector{T}, x0::Vector{T}, h::Float64, coeffs::Vector{Float64}, k::Vector{Vector{T}})
    @inbounds for m=1:length(x0)
        dx::T = 0.
        @inbounds for i=1:length(coeffs)
            dx += coeffs[i]::Float64*k[i][m]
        end
        x[m] = x0[m] + h*dx
    end
    return nothing
end

function dopri_step3{T}(F::Function, t::Float64, h::Float64,
                x0::Vector{T}, xp::Vector{T}, xs::Vector{T}, k::Vector{Vector{T}})
    for i=2:length(c)
        substep2(xp, x0, h, a[i], k)
        F(t + h*c[i], xp, k[i])
    end
    substep2(xs, x0, h, bs, k)
    return nothing
end



function step1{T}(xp::Vector{T}, x::Vector{T}, h::Float64, k1::Vector{T})
    @inbounds for m=1:length(x)
        xp[m] = x[m] + h*a2[1]*k1[m]
    end
end

function step2{T}(xp::Vector{T}, x::Vector{T}, h::Float64, k1::Vector{T}, k2::Vector{T})
    @inbounds for m=1:length(x)
        dx = a3[1]*k1[m] + a3[2]*k2[m]
        xp[m] = x[m] + h*dx
    end
end

function step3{T}(xp::Vector{T}, x::Vector{T}, h::Float64, k1::Vector{T}, k2::Vector{T}, k3::Vector{T})
    @inbounds for m=1:length(x)
        dx = a4[1]*k1[m] + a4[2]*k2[m] + a4[3]*k3[m]
        xp[m] = x[m] + h*dx
    end
end

function step4{T}(xp::Vector{T}, x::Vector{T}, h::Float64, k1::Vector{T}, k2::Vector{T}, k3::Vector{T}, k4::Vector{T})
    @inbounds for m=1:length(x)
        dx = a5[1]*k1[m] + a5[2]*k2[m] + a5[3]*k3[m] + a5[4]*k4[m]
        xp[m] = x[m] + h*dx
    end
end

function step5{T}(xp::Vector{T}, x::Vector{T}, h::Float64, k1::Vector{T}, k2::Vector{T}, k3::Vector{T}, k4::Vector{T}, k5::Vector{T})
    @inbounds for m=1:length(x)
        dx = a6[1]*k1[m] + a6[2]*k2[m] + a6[3]*k3[m] + a6[4]*k4[m]
        dx += a6[5]*k5[m]
        xp[m] = x[m] + h*dx
    end
end

function step6{T}(xp::Vector{T}, x::Vector{T}, h::Float64, k1::Vector{T}, k2::Vector{T}, k3::Vector{T}, k4::Vector{T}, k5::Vector{T}, k6::Vector{T})
    @inbounds for m=1:length(x)
        dx = a7[1]*k1[m] + a7[3]*k3[m] + a7[4]*k4[m]
        dx += a7[5]*k5[m] + a7[6]*k6[m]
        xp[m] = x[m] + h*dx
    end
end

function error_estimates{T}(xp::Vector{T}, x::Vector{T}, h::Float64, k1::Vector{T}, k2::Vector{T}, k3::Vector{T}, k4::Vector{T}, k5::Vector{T}, k6::Vector{T})
    max_difference::Float64 = 0.
    max_value::Float64 = 0.
    @inbounds for m=1:length(x)
        dx = bs[1]*k1[m] + bs[3]*k3[m] + bs[4]*k4[m]
        dx += bs[5]*k5[m] + bs[6]*k6[m] + bs[7]*k7[m]
        v = abs2(x[m] + h*dx - xp[m])
        #v = abs2(xs[m]-xp[m])
        if v > max_difference
            max_difference = v
        end
        v = abs2(xp[m])
        if v > max_value
            max_value = v
        end
    end
    # println("DiffValue: ", sqrt(max_difference))
    # println("MValue: ", sqrt(max_value))
    # println("|xp|", norm(xp,Inf))
    # println("|x|", norm(x,Inf))
    return sqrt(max_value), sqrt(max_difference)
end

function dopri_step{T}(F::Function, t::Float64, h::Float64,
                x::Vector{T}, xp::Vector{T},
                k1::Vector{T}, k2::Vector{T}, k3::Vector{T}, k4::Vector{T}, k5::Vector{T}, k6::Vector{T}, k7::Vector{T})
    step1(xp, x, h, k1)
    F(t + h*c[2], xp, k2)
    step2(xp, x, h, k1, k2)
    F(t + h*c[3], xp, k3)
    step3(xp, x, h, k1, k2, k3)
    F(t + h*c[4], xp, k4)
    step4(xp, x, h, k1, k2, k3, k4)
    F(t + h*c[5], xp, k5)
    step5(xp, x, h, k1, k2, k3, k4, k5)
    F(t + h*c[6], xp, k6)
    step6(xp, x, h, k1, k2, k3, k4, k5, k6)
    F(t + h*c[7], xp, k7)
    return error_estimates(xp, x, h, k1, k2, k3, k4, k5, k6)
end


function dopri_step2{T}(F::Function, t::Float64, h::Float64,
                x::Vector{T}, xp::Vector{T},
                k1::Vector{T}, k2::Vector{T}, k3::Vector{T}, k4::Vector{T}, k5::Vector{T}, k6::Vector{T}, k7::Vector{T})
    # step 2
    #dim = length(x)
    @inbounds for m=1:length(x)
        xp[m] = x[m] + h*a2[1]*k1[m]
    end
    F(t + h*c[2], xp, k2)
    # step 3
    @inbounds for m=1:length(x)
        dx = a3[1]*k1[m] + a3[2]*k2[m]
        xp[m] = x[m] + h*dx
    end
    F(t + h*c[3], xp, k3)
    # step 4
    @inbounds for m=1:length(x)
        dx = a4[1]*k1[m] + a4[2]*k2[m] + a4[3]*k3[m]
        xp[m] = x[m] + h*dx
    end
    F(t + h*c[4], xp, k4)
    # step 5
    @inbounds for m=1:length(x)
        dx = a5[1]*k1[m] + a5[2]*k2[m] + a5[3]*k3[m] + a5[4]*k4[m]
        xp[m] = x[m] + h*dx
    end
    F(t + h*c[5], xp, k5)
    # step 6
    @inbounds for m=1:length(x)
        dx = a6[1]*k1[m] + a6[2]*k2[m] + a6[3]*k3[m] + a6[4]*k4[m]
        dx += a6[5]*k5[m]
        xp[m] = x[m] + h*dx
    end
    F(t + h*c[6], xp, k6)
    # step 7
    @inbounds for m=1:length(x)
        dx = a7[1]*k1[m] + a7[3]*k3[m] + a7[4]*k4[m]
        dx += a7[5]*k5[m] + a7[6]*k6[m]
        xp[m] = x[m] + h*dx
    end
    F(t + h*c[7], xp, k7)
    # Final
    max_difference::Float64 = 0.
    max_value::Float64 = 0.
    @inbounds for m=1:length(x)
        dx = bs[1]*k1[m] + bs[3]*k3[m] + bs[4]*k4[m]
        dx += bs[5]*k5[m] + bs[6]*k6[m] + bs[7]*k7[m]
        v = abs2(x[m] + h*dx - xp[m])
        if v > max_difference
            max_difference = v
        end
        v = abs2(xp[m])
        if v > max_value
            max_value = v
        end
    end
    # println("DiffValue: ", sqrt(max_difference))
    # println("MValue: ", sqrt(max_value))
    # println("|xp|", norm(xp,Inf))
    # println("|x|", norm(x,Inf))
    return sqrt(max_value), sqrt(max_difference)
end


function f(t::Float64, x::Vector{Complex128}, result::Vector{Complex128})
    for i=1:length(x)
        result[i] = cos(x[i]*t)
    end
end

const N = 800
srand(0)
const x0 = complex(rand(N^2))#Complex128[1., 0., 0. ,0.]
const x = zeros(Complex128, N^2)
const xs = zeros(Complex128, N^2)
const xp = zeros(Complex128, N^2)
const k1 = zeros(Complex128, N^2)
const k2 = zeros(Complex128, N^2)
const k3 = zeros(Complex128, N^2)
const k4 = zeros(Complex128, N^2)
const k5 = zeros(Complex128, N^2)
const k6 = zeros(Complex128, N^2)
const k7 = zeros(Complex128, N^2)



function g(t::Complex128, x::Matrix{Complex128}, result::Matrix{Complex128})
    for j=1:size(x,2), i=1:size(x,1)
        result[i,j] = cos(x[i,j]*t)
    end
end

const x0_ = reshape(deepcopy(x0), N, N)
const x_ = zeros(Complex128, N, N)
const dx_ = zeros(Complex128, N, N)
const tmp_ = zeros(Complex128, N, N)
const xs_ = zeros(Complex128, N, N)
const xp_ = zeros(Complex128, N, N)
const k1_ = zeros(Complex128, N, N)
const k2_ = zeros(Complex128, N, N)
const k3_ = zeros(Complex128, N, N)
const k4_ = zeros(Complex128, N, N)
const k5_ = zeros(Complex128, N, N)
const k6_ = zeros(Complex128, N, N)
const k7_ = zeros(Complex128, N, N)


const k_ = Matrix{Complex128}[k1_, k2_, k3_, k4_, k5_, k6_, k7_]

function profile_optimized(runs)
    t = 1.
    h = 0.1
    f(t,x0,k1)
    for i=1:runs
        dopri_step(f, t, h,
                    x0, xp,
                    k1, k2, k3, k4, k5, k6, k7)
        #println("regular |xp|: ", norm(xp,Inf))
        t += h
    end
end

k = Vector{Complex128}[]
push!(k, k1)
push!(k, k2)
push!(k, k3)
push!(k, k4)
push!(k, k5)
push!(k, k6)
push!(k, k7)

function profile_optimized2(runs)
    t = 1.
    h = 0.1
    f(t,x0,k1)
    for i=1:runs
        dopri_step3(f, t, h,
                    x0, xp, xs,
                    k)
        #println("regular |xp|: ", norm(xp,Inf))
        t += h
    end
end

function profile(runs)
    t = 1.0+0.0im
    h = 0.1+0.0im
    g(t,x0_,k1_)
    for i=1:runs
        oderk_step(g, t, h,
                    x0_, a_compl, bs_compl, bp_compl, c_compl,
                    xs_, xp_, dx_, tmp_,
                    k_)
        #println("regular |xp_|: ", norm(reshape(xp_,N^2),Inf))
        t += h
    end
    return nothing
end

const runs = 1

println("=======Regular========")
@time profile(1)
@time profile(runs)

println("=======Optimized========")
@time profile_optimized2(1)
@time profile_optimized2(runs)

# println("=======Optimized2========")
# @time profile_optimized(1)
# @time profile_optimized(runs)

println(norm(xp-reshape(xp_,N^2)))
