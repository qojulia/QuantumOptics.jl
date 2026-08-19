function savebenchmark(name, Ncutoff, T)
    results = '[';
    for i = 1:length(Ncutoff)
        results = [results, '{"N": ', mat2str(Ncutoff(i)), ', "t": ', mat2str(T(i)), '}'];
        if i == length(Ncutoff)
            results = [results, ']'];
        else
            results = [results, ', '];
        end
    end
    fid = fopen(['../results/results-QuantumOpticsToolbox-', name, '.json'], 'wt');
    fprintf(fid, results);
    fclose(fid);
end