
function results = bench_multiplication_sparse_dense()
    basename = 'multiplication_sparse_dense';
    cutoffs = 50:50:1000;
    S = [0.1, 0.01, 0.001];
    Nrand = 5;
    for s = S
        name = [basename, '_', strrep(mat2str(s), '.', '')];
        results = [];
        for N = cutoffs
            T = 0.;
            for i=1:Nrand
                [op1, op2] = setup(N, s);
                f_ = @() f(op1, op2);
                T = T + timeit(f_);
            end
            results = [results, T/Nrand];
        end
        savebenchmark(name, cutoffs, results)
    end
end

function [op1, op2] = setup(N, s)
    op1 = (1.+0.3j)*sprand(N, N, s);
    op2 = (1.+0.3j)*rand(N, N);
end

function result = f(op1, op2)
    result = op1*op2;
end
