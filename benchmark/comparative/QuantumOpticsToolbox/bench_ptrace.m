
function result = bench_ptrace()
    name = 'ptrace';
    cutoffs = [2:9];
    checks = [];
    result = [];
    for N = cutoffs
        op = create_operator(N);
        f_ = @() f(op);
        y = f_();
        checks = [checks, sumabs(y.data)];
        result = [result, timeit(f_)];
    end
    checkbenchmark(name, cutoffs, checks, 1e-5)
    savebenchmark(name, 4*cutoffs.^2, result)
end

function result = f(op)
    result = ptrace(op, [2, 3]);
end

function result = create_operator(Ncutoff)
    op1 = create_suboperator(1, 0.2, Ncutoff);
    op2 = create_suboperator(-2, 0.3, Ncutoff);
    op3 = create_suboperator(3, 0.4, 2);
    op4 = create_suboperator(4, 0.5, 2);
    result = tensor(op1, op2, op3, op4);
end

function result = create_suboperator(c0, alpha, N)
    x = linspace(0., 1., N^2);
    data = reshape(c0 + alpha*x, N, N);
    result = qo(data);
end