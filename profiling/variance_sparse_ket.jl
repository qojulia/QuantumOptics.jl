using BenchmarkTools
using QuantumOptics

srand(0)
randstate(b) = normalize(Ket(b, rand(Complex128, length(b))))

function f1(op::SparseOperator, state::Ket)
    expect(op^2, state) - expect(op, state)^2
end

function f2(op::SparseOperator, state::Ket)
    x = op*state
    stateT = dagger(state)
    stateT*op*x - (stateT*x)^2
end

N = 1000
b = GenericBasis(N)
s = 0.1
op = SparseOperator(b, sprand(Complex128, N, N, s))
state = randstate(b)

println(f1(op, state))
println(f2(op, state))

function run_f1(N::Int, op, state)
    for i=1:N
        f1(op, state)
    end
end

function run_f2(N::Int, op, state)
    for i=1:N
        f2(op, state)
    end
end

@time f1(op, state)
@time f1(op, state)
@time f2(op, state)
@time f2(op, state)

Profile.clear()
# @profile run_f1(100000, op, state)
# @profile run_f2(100000, op, state)

r2 = @benchmark f2($op, $state)
r1 = @benchmark f1($op, $state)

println(r1)
println(r2)
