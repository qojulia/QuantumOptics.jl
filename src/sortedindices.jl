module sortedindices

"""
6, [1, 4] => [2, 3, 5, 6]
"""
function complement(N::Int, indices::Vector{Int})
    L = length(indices)
    x = Vector{Int}(N - L)
    i_ = 1 # Position in the x vector
    j = 1 # Position in indices vector
    for i=1:N
        if j > L || indices[j]!=i
            x[i_] = i
            i_ += 1
        else
            j += 1
        end
    end
    x
end

"""
[1, 4, 5], [2, 4, 7] => [1, 2, 4, 5, 7]
"""
function union(ind1::Vector{Int}, ind2::Vector{Int})
    i1 = 1
    i2 = 1
    N1 = length(ind1)
    N2 = length(ind2)
    xvec = Vector{Int}()
    while true
        if i1 > N1
            for j=i2:N2
                push!(xvec, ind2[j])
            end
            return xvec
        elseif i2 > N2
            for j=i1:N1
                push!(xvec, ind1[j])
            end
            return xvec
        end
        x1 = ind1[i1]
        x2 = ind2[i2]
        if x1 == x2
            i1 += 1
            i2 += 1
            push!(xvec, x1)
        elseif x1 < x2
            i1 += 1
            push!(xvec, x1)
        else
            i2 += 1
            push!(xvec, x2)
        end
    end
end


"""
[1, 4, 5], [2, 4, 7] => [1, 5]
"""
function remove(ind1::Vector{Int}, ind2::Vector{Int})
    x = Int[]
    for i in ind1
        if i ∉ ind2
            push!(x, i)
        end
    end
    x
end

"""
[1, 4, 5], [2, 4, 7] => [1, 3]
"""
function shiftremove(ind1::Vector{Int}, ind2::Vector{Int})
    x = Int[]
    for i in ind1
        if i ∉ ind2
            counter = 0
            for i2 in ind2
                if i2 < i
                    counter += 1
                else
                    break
                end
            end
            push!(x, i-counter)
        end
    end
    x
end

"""
[2, 3, 6], [1, 3, 4, 6, 7] => [3, 6]
"""
function intersect(ind1::Vector{Int}, ind2::Vector{Int})
    i1 = 1
    i2 = 1
    N1 = length(ind1)
    N2 = length(ind2)
    xvec = Vector{Int}()
    if i1 > N1 || i2 > N2
        return xvec
    end
    x1 = ind1[i1]
    x2 = ind2[i2]
    while true
        if x1 == x2
            i1 += 1
            i2 += 1
            push!(xvec, x1)
            if i1 > N1 || i2 > N2
                return xvec
            end
            x1 = ind1[i1]
            x2 = ind2[i2]
        elseif x1 < x2
            i1 += 1
            if i1 > N1
                return xvec
            end
            x1 = ind1[i1]
        else
            i2 += 1
            if i2 > N2
                return xvec
            end
            x2 = ind2[i2]
        end
    end
end

function reducedindices(I_::Vector{Int}, I::Vector{Int})
    N = length(I_)
    x = Vector{Int}(N)
    for n in 1:N
        x[n] = findfirst(I, I_[n])
    end
    x
end

function reducedindices!(I_::Vector{Int}, I::Vector{Int})
    for n in 1:length(I_)
        I_[n] = findfirst(I, I_[n])
    end
end

end # module
