function checkbenchmark(name, cutoffs, results, eps)
    [cutoffs_f, results_f] = loadjson(name);
    for i=1:length(cutoffs)
        Ncutoff = cutoffs(i);
        index = find(cutoffs_f==Ncutoff, 1);
        if isempty(index)
            warning(['Cutoff of ', num2str(Ncutoff), ' not found in check file.'])
        else
            r1 = results_f(index);
            r2 = results(i);
            abs(r1-r2);
            if (abs(r1-r2)/abs(r1) > eps)
                warning(['Result may be incorrect: ', num2str(r2), ' instead of ', num2str(r1)])
            end
        end
    end
end

function [cutoffs, results] = loadjson(name)
    fid = fopen(['../checks/', name, '.json'], 'r');
    data = fscanf(fid, '%c');
    data = data(2:end-1);
    fclose(fid);
    results = [];
    cutoffs = [];
    for item=strsplit(data, ',')
        x = strsplit(cell2mat(item), ':');
        N = cell2mat(x(1));
        check = cell2mat(x(2));
        N = N(2:end-1);
        cutoffs = [cutoffs; str2num(N)];
        results = [results; str2num(check)];
    end
    [cutoffs, I] = sort(cutoffs);
    results = results(I);
end
