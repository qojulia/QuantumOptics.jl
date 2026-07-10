
function result = bench_expect_operator()
    name = 'expect_operator';
    cutoffs = [100:100:700];
    checks = [];
    result = [];
    for N = cutoffs
        [op, rho] = setup(N);
        f_ = @() f(op, rho);
        checks = [checks, f_()];
        result = [result, timeit(f_)];
    end
    checkbenchmark(name, cutoffs, checks, 1e-5)
    savebenchmark(name, cutoffs, result)
end

function [op, rho] = setup(N)
    op = (destroy(N) + create(N));
    psi = qo(ones(N, 1)/(N^(1/2)));
    rho = psi*psi';
end

function result = f(op, state)
    result = expect(op, state);
end
