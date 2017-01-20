require("../src/quantumoptics.jl")

using QuantumOptics

srand(0)

N1 = 2
N2 = 2
N3 = 2
N4 = 2
N5 = 4
N6 = 4
N7 = 32
N = N1*N2*N3*N4*N5*N6*N7

b1 = GenericBasis([N1])
b2 = GenericBasis([N2])
b3 = GenericBasis([N3])
b4 = GenericBasis([N4])
b5 = GenericBasis([N5])
b6 = GenericBasis([N6])
b7 = GenericBasis([N7])

basis_total = CompositeBasis(b1, b2, b3, b4, b5, b6); N = N1*N2*N3*N4*N5*N6
#basis_total = CompositeBasis(b1, b2, b3, b4, b5, b6, b7)

op1 = Operator(b1, rand(N1,N1))
op2 = Operator(b2, rand(N2,N2))

psi = Ket(basis_total, rand(N))

psi2 = Ket(basis_total, rand(N))
psi = tensor(psi, dagger(psi))
@time psi2 = tensor(psi2, dagger(psi2))

l1 = operators_lazy.LazyTensor(basis_total, basis_total, [1], [op1])
l2 = operators_lazy.LazyTensor(basis_total, basis_total, [2], [op2])
l3 = operators_lazy.LazyTensor(basis_total, basis_total, [1,2], [op1,op2])

#@time operators_lazy.gemm2!(Complex(1.), l3, psi.data, Complex(0.), psi2.data)
#@time operators_lazy.gemm2!(Complex(1.), l3, psi.data, Complex(0.), psi2.data)
#println(vecnorm(vec((psi2-full(l3)*psi).data)))
@time operators_lazy.gemm!(Complex(1.), l3, psi.data, Complex(0.), psi2.data)
@time operators_lazy.gemm!(Complex(1.), l3, psi.data, Complex(0.), psi2.data)
println(vecnorm(vec((psi2-full(l3)*psi).data)))

# @time operators_lazy.gemm2!(Complex(1.), psi.data, l3, Complex(0.), psi2.data)
# @time operators_lazy.gemm2!(Complex(1.), psi.data, l3, Complex(0.), psi2.data)
# println(vecnorm(vec((psi2-psi*full(l3)).data)))
@time operators_lazy.gemm!(Complex(1.), psi.data, l3, Complex(0.), psi2.data)
@time operators_lazy.gemm!(Complex(1.), psi.data, l3, Complex(0.), psi2.data)
println(vecnorm(vec((psi2-psi*full(l3)).data)))

# @time operators_lazy.gemm!(Complex(1.), (l2+l3), psi.data, Complex(0.), psi2.data)
# @time operators_lazy.gemm!(Complex(1.), (l2+l3), psi.data, Complex(0.), psi2.data)
# println(vecnorm(vec(((l3+l2)*psi - psi2).data)))

# @time operators_lazy.gemm!(Complex(1.), psi.data, (l2+l3), Complex(0.), psi2.data)
# @time operators_lazy.gemm!(Complex(1.), psi.data, (l2+l3), Complex(0.), psi2.data)
# println(vecnorm(vec((psi*(l3+l2) - psi2).data)))
# println(vecnorm(vec(((l3+l2)*psi - l3*psi - l2*psi).data)))



