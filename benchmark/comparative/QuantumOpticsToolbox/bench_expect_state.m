
function result = bench_expect_state()
    name = 'expect_state';
    cutoffs = [5000:5000:70000];
    checks = [];
    result = [];
    for N = cutoffs
        [op, psi] = setup(N);
        f_ = @() f(op, psi);
        checks = [checks, f_()];
        result = [result, timeit(f_)];
    end
    checkbenchmark(name, cutoffs, checks, 1e-5)
    savebenchmark(name, cutoffs, result)
end

function [op, psi] = setup(N)
    op = (destroy(N) + create(N));
    psi = qo(ones(N, 1)/(N^(1/2)));
end

function result = f(op, state)
    result = expect(op, state);
end
