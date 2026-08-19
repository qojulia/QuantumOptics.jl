
function result = bench_timeevolution_master_particle()
    name = 'timeevolution_master_particle';
    cutoffs = 5:5:30;
    checks = [];
    result = [];
    for N = cutoffs
        [rho0, L, x] = setup(N);
        f_ = @() f(rho0, L, x);
        checks = [checks, sum(f_())];
        result = [result, timeit(f_)];
    end
    checkbenchmark(name, cutoffs, checks, 1e-5)
    savebenchmark(name, cutoffs, result)
end

function [rho0, L, x] = setup(N)
    xmin = -5;
    xmax = 5;
    x0 = 0.3;
    p0 = -0.2;
    sigma0 = 1;
    dx = (xmax - xmin)/N;
    pmin = -pi/dx;
    pmax = pi/dx;
    dp = (pmax - pmin)/N;
    tmp = linspace(xmin, xmax, N+1);
    samplepoints_x = tmp(1:end-1);
    tmp = linspace(pmin, pmax, N+1);
    samplepoints_p = tmp(1:end-1);

    x = qo(diag(samplepoints_x));

    % Create momentum operator
    row0 = zeros(1, N);
    for i=1:N
        dxji = samplepoints_x(i) - xmin;
        for j=1:N
            p = samplepoints_p(j);
            row0(i) = row0(i) + p*exp(-1j*p*dxji)*dp*dx/(2*pi);
        end
    end
    col0 = conj(row0);
    a = zeros(N, N);
    for i=1:N
        a(i, i:end) = row0(1:N-i+1);
        a(i:end, i) = col0(1:N-i+1);
    end
    p = qo(a);

    H = p^2 + 2*x^2;

    % Create gaussian state
    alpha = 1./(pi^(1/4)*sqrt(sigma0))*sqrt(dx);
    data = alpha*exp(1j*p0*(samplepoints_x-x0/2) - (samplepoints_x-x0).^2/(2*sigma0^2));
    psi0 = qo(data.');
    C = (x + 1j*p);
    LH = -1i * (spre(H) - spost(H));
    L1 = spre(C)*spost(C')-0.5*spre(C'*C)-0.5*spost(C'*C);
    L = LH+L1;

    rho0 = psi0 * psi0';
end

function result = f(rho0, L, x)
    % Set up options, if required
    options.reltol = 1e-6;
    options.abstol = 1e-8;
    tlist = linspace(0, 10, 11);
    ode2file('file1.dat', L, rho0, tlist, options);
    % Call the equation solver
    odesolve('file1.dat','file2.dat');
    % Read in the output data file
    fid = fopen('file2.dat','rb');
    rho = qoread(fid, dims(rho0), size(tlist));
    fclose(fid);
    result = real(expect(x, rho));
end
