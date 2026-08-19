
function result = bench_timeevolution_master_cavity()
    name = 'timeevolution_master_cavity';
    cutoffs = 10:10:60;
    checks = [];
    result = [];
    for N = cutoffs
        f_ = @() f(N);
        checks = [checks, sum(f_())];
        result = [result, timeit(f_)];
    end
    checkbenchmark(name, cutoffs, checks, 1e-5)
    savebenchmark(name, cutoffs, result)
end

function result = f(Ncutoff)
    kappa = 1.;
    eta = 1.5;
    wc = 1.8;
    wl = 2;
    delta_c = wl - wc;
    alpha0 = 0.3 - 0.5j;
    tspan = 0:1:10;

    a = destroy(Ncutoff);
    at = create(Ncutoff);
    n = at*a;

    H = delta_c*n + eta*(a + at);
    C = sqrt(kappa)*a;
    LH = -1i * (spre(H) - spost(H));
    L1 = spre(C)*spost(C')-0.5*spre(C'*C)-0.5*spost(C'*C);
    L = LH+L1;

    D = expm(alpha0*at - conj(alpha0)*a);
    psi0 = D*basis(Ncutoff, 1);
    expect(a, psi0)
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
    result = real(expect(n, rho));
end
