
function results = bench_multiplication_dense_dense()
    name = 'multiplication_dense_dense';
    cutoffs = [50:50:1001];
    results = [];
    for N = cutoffs
        [op1, op2] = setup(N);
        f_ = @() f(op1, op2);
        results = [results, timeit(f_)];
    end
    savebenchmark(name, cutoffs, results)
end

function [op1, op2] = setup(N)
    op1 = (1.+0.3j)*rand(N, N);
    op2 = (1.+0.3j)*rand(N, N);
end

function result = f(op1, op2)
    result = op1*op2;
end
