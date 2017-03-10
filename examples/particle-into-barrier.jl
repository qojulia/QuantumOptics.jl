
using QuantumOptics
using PyPlot

xmin = -30
xmax = 30
Npoints = 200

b_position = PositionBasis(xmin, xmax, Npoints)
b_momentum = MomentumBasis(b_position);

V0 = 1. # Height of Barrier
d = 5 # Width of Barrier
function V_barrier(x)
    if x < -d/2 || x > d/2
        return 0.
    else
        return V0
    end
end
V = potentialoperator(b_position, V_barrier);

Txp = FFTOperator(b_position, b_momentum)
Tpx = FFTOperator(b_momentum, b_position)
Hkin = LazyProduct(Txp, momentumoperator(b_momentum)^2/2, Tpx);

H = LazySum(Hkin, V);

xpoints = samplepoints(b_position)

x0 = -15
sigma0 = 4
p0vec = [sqrt(0.1), 1, sqrt(2), sqrt(3), 2]
timecuts = 20

for i_p in 1:length(p0vec)
    p0 = p0vec[i_p]
    Ψ₀ = gaussianstate(b_position, x0, p0, sigma0)
    scaling = 1./maximum(abs(Ψ₀.data))^2/5
    n0 = abs(Ψ₀.data).^2*scaling

    tmax = 2*abs(x0)/(p0+0.2)
    T = collect(linspace(0., tmax, timecuts))
    tout, Ψt = timeevolution.schroedinger(T, Ψ₀, H);

    offset = real(expect(Hkin, Ψ₀))
    plot(xpoints, n0+offset, "C$i_p--")
    for i=1:length(T)
        Ψ = Ψt[i]
        n = abs(Ψ.data).^2*scaling
        plot(xpoints, n+offset, "C$i_p", alpha=0.3)
    end
    nt = abs(Ψt[timecuts].data).^2*scaling
    plot(xpoints, nt+offset, "C$i_p")
end
y = V_barrier.(xpoints)
plot(xpoints, y, "k")
fill_between(xpoints, 0, y, color="k", alpha=0.5);


