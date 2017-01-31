using Base.Test
using QuantumOptics

srand(0)

alpha = 0.5

fockbasis = FockBasis(2)
spinbasis = SpinBasis(1//2)

a = destroy(fockbasis)
at = create(fockbasis)
n = number(fockbasis)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)

Ispin = sparse_identityoperator(spinbasis)
Ifock = sparse_identityoperator(fockbasis)

# LazyTensor
basis = tensor(spinbasis, fockbasis, spinbasis)
psi_ket = tensor(basis_ket(spinbasis, 2), coherentstate(fockbasis, alpha), basis_ket(spinbasis, 1))
psi_bra = dagger(psi_ket)

op1 = LazyTensor(basis, basis, 2, a)
op2 = LazyTensor(basis, basis, 3, sy)
op3 = LazyTensor(basis, basis, [2,3], [a,sy])
op4 = LazyTensor(basis, basis, [1,2,3], [sx,a,sy])

result_ket = deepcopy(psi_ket)
result_bra = deepcopy(psi_bra)

@test Ispin ⊗ a ⊗ Ispin == sparse(op1)
@test Ispin ⊗ Ifock ⊗ sy == sparse(op2)
@test Ispin ⊗ a ⊗ sy == sparse(op3)
@test sx ⊗ a ⊗ sy == sparse(op4)

@test full(Ispin ⊗ a ⊗ Ispin) == full(op1)
@test full(Ispin ⊗ Ifock ⊗ sy) == full(op2)
@test full(Ispin ⊗ a ⊗ sy) == full(op3)
@test full(sx ⊗ a ⊗ sy) == full(op4)

for op=[op1, op2, op3, op4]
    @test typeof(op) == LazyTensor
    operators.gemv!(Complex(1.), op, psi_ket, Complex(0.), result_ket)
    @test_approx_eq_eps 0. norm(full(op)*psi_ket - result_ket) 1e-12
    operators.gemv!(Complex(1.), psi_bra, op, Complex(0.), result_bra)
    @test_approx_eq_eps 0. norm(psi_bra*full(op) - result_bra) 1e-12
end


# LazySum
psi_ket = coherentstate(fockbasis, alpha)
psi_bra = dagger(psi_ket)
op = LazySum(a, at)
@test typeof(op) == LazySum
@test (a+at) == sparse(op)
@test full(a+at) == full(op)

@test_approx_eq_eps 0. norm(op*psi_ket - (a+at)*psi_ket) 1e-12

psi_ket2 = Ket(fockbasis)
operators.gemv!(Complex(1.), LazySum(a, at, at*a), psi_ket, Complex(0.), psi_ket2)
@test_approx_eq_eps 0. norm((a+at+at*a)*psi_ket - psi_ket2) 1e-12

psi_bra2 = Bra(fockbasis)
operators.gemv!(Complex(1.), psi_bra, LazySum(a, at, at*a), Complex(0.), psi_bra2)
@test_approx_eq_eps 0. norm(psi_bra*(a+at+at*a) - psi_bra2) 1e-12


# LazyProduct
n_lazy = LazyProduct(at, a)
@test_approx_eq_eps 0. norm(n_lazy*psi_ket - n*psi_ket) 1e-12
@test_approx_eq_eps 0. norm(psi_bra*n_lazy - psi_bra*n) 1e-12

zero_op = n_lazy - n
@test typeof(zero_op) == LazySum
@test (at*a) == sparse(n_lazy)
@test full(at*a) == full(n_lazy)

@test_approx_eq_eps 0. norm(zero_op*psi_ket) 1e-12
@test_approx_eq_eps 0. norm(psi_bra*zero_op) 1e-12

psi_ket_ = deepcopy(psi_ket)
psi_ket2 = Ket(fockbasis)
operators.gemv!(Complex(1.), LazyProduct(a, at, at*a), psi_ket_, Complex(0.), psi_ket2)
@test_approx_eq_eps 0. norm(a*at*at*a*psi_ket - psi_ket2) 1e-12

psi_bra_ = deepcopy(psi_bra)
psi_bra2 = Bra(fockbasis)
operators.gemv!(Complex(1.), psi_bra_, LazyProduct(a, at, at*a), Complex(0.), psi_bra2)
@test_approx_eq_eps 0. norm(psi_bra*a*at*at*a - psi_bra2) 1e-12


# LazySum of LazyTensor
basis = tensor(spinbasis, fockbasis)
psi_ket = tensor(basis_ket(spinbasis, 2), coherentstate(fockbasis, alpha))
psi_bra = dagger(psi_ket)

op1 = LazyTensor(basis, basis, 1, sy)
op2 = LazyTensor(basis, basis, 2, a)
op3 = LazyTensor(basis, basis, [1,2], [sx+sy, a])

op = LazySum(op1, op2, op3)

result_ket = deepcopy(psi_ket)
result_ket2 = deepcopy(psi_ket)
result_bra = deepcopy(psi_bra)

@test sy ⊗ Ifock == sparse(op1)
@test Ispin ⊗ a == sparse(op2)
@test (sx+sy) ⊗ a == sparse(op3)
@test sy ⊗ Ifock + Ispin ⊗ a + (sx+sy) ⊗ a == sparse(op)

@test full(sy ⊗ Ifock) == full(sparse(op1))
@test full(Ispin ⊗ a) == full(sparse(op2))
@test full((sx+sy) ⊗ a) == full(sparse(op3))
@test full(sy ⊗ Ifock + Ispin ⊗ a + (sx+sy) ⊗ a) == full(sparse(op))

op = LazySum(sparse(op1))

operators.gemv!(Complex(1.), op2, psi_ket, Complex(0.), result_ket)
operators.gemv!(Complex(1.), sparse(op2), psi_ket, Complex(0.), result_ket2)
# operators.gemv!(Complex(1.), op1, psi_ket, Complex(1.), result_ket)
@test_approx_eq_eps 0. norm(result_ket2 - result_ket) 1e-12

@test typeof(op) == LazySum
operators.gemv!(Complex(1.), op, psi_ket, Complex(0.), result_ket)
@test_approx_eq_eps 0. norm(full(op)*psi_ket - result_ket) 1e-12
operators.gemv!(Complex(1.), psi_bra, op, Complex(0.), result_bra)
@test_approx_eq_eps 0. norm(psi_bra*full(op) - result_bra) 1e-12
