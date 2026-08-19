
function result = bench_timeevolution_master_jaynescummings()
    name = 'timeevolution_master_jaynescummings';
    cutoffs = 5:5:60;
    checks = [];
    result = [];
    for N = cutoffs
        f_ = @() f(N);
        checks = [checks, sum(f_())];
        result = [result, timeit(f_)];
    end
    checkbenchmark(name, 2*cutoffs, checks, 1e-5)
    savebenchmark(name, 2*cutoffs, result)
end

function result = f(Ncutoff)
    wa = 1;
    wc = 1;
    g = 2;
    kappa = 0.5;
    gamma = 0.1;
    n_th = 0.75;
    tspan = 0:1:10;

    Ia = identity(2);
    Ic = identity(Ncutoff);

    a = destroy(Ncutoff);
    at = a';
    n = at*a;

    sm = sigmam();
    sp = sigmap();
    sz = sigmaz();

    H = tensor(n, wc*Ia) + tensor(Ic, wa/2. *sz) + g*(tensor(at, sm) + tensor(a, sp));
    C1 = tensor(sqrt(kappa*(1+n_th))*a, Ia);
    C2 = tensor(sqrt(kappa*n_th)*at, Ia);
    C3 = tensor(Ic, sqrt(gamma)*sm);
    LH = -1i * (spre(H) - spost(H));
    L1 = spre(C1)*spost(C1')-0.5*spre(C1'*C1)-0.5*spost(C1'*C1);
    L2 = spre(C2)*spost(C2')-0.5*spre(C2'*C2)-0.5*spost(C2'*C2);
    L3 = spre(C3)*spost(C3')-0.5*spre(C3'*C3)-0.5*spost(C3'*C3);
    L = LH + L1 + L2 + L3;

    psi0 = tensor(basis(Ncutoff, 1), (basis(2, 1) + basis(2, 2))/sqrt(2));
    rho0 = psi0 * psi0';

    % Set up options, if required
    options.reltol = 1e-6;
    options.abstol = 1e-8;
    % Write out the data file
    ode2file('file1.dat', L, rho0, tspan, options);
    % Call the equation solver
    odesolve('file1.dat','file2.dat');
    % Read in the output data file
    fid = fopen('file2.dat','rb');
    rho = qoread(fid, dims(rho0), size(tspan));
    fclose(fid);
    result = real(expect(tensor(n, Ia), rho));
end
