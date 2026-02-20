

"""
    timeevolution.master_h_heisenberg(tspan, A0, H, J; <keyword arguments>)

Integrate the heisenberg picture master equation with dmaster_h_heisenberg as derivative function.
See [`master_h`](@ref) for the Schrödinger picture.
"""
function master_h_heisenberg(tspan, A0::Operator, H::AbstractOperator, J;
                           rates=nothing,
                           Jdagger=dagger.(J),
                           fout=nothing,
                           kwargs...)
    _check_const(H)
    _check_const.(J)
    _check_const.(Jdagger)
    tspan, A0 = _promote_time_and_state(A0, H, J, tspan)
    tmp = copy(A0)
    dmaster_heisenberg_(t, A, dA) = dmaster_h_heisenberg!(dA, H, J, Jdagger, rates, A, tmp)
    integrate_master(tspan, dmaster_heisenberg_, A0, fout; kwargs...)
end



"""
    dmaster_h_heisenberg!(dA, H, J, Jdagger, rates::Nothing, A, dA_cache)

Update `drho` according to a heisenberg picture Lindblad equation.
See [`dmaster_h`](@ref).
"""
function dmaster_h_heisenberg!(dA, H, J, Jdagger, rates::Nothing, A, dA_cache)
    QuantumOpticsBase.mul!(dA,H,A,eltype(A)(im),zero(eltype(A)))
    QuantumOpticsBase.mul!(dA,A,H,-eltype(A)(im),one(eltype(A)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(dA_cache,Jdagger[i],A)
        QuantumOpticsBase.mul!(dA,dA_cache,J[i],true,true)

        QuantumOpticsBase.mul!(dA_cache,Jdagger[i],J[i],eltype(A)(-0.5),zero(eltype(A)))
        QuantumOpticsBase.mul!(dA,dA_cache,A,true,true)
        QuantumOpticsBase.mul!(dA,A,dA_cache,true,true)
    end
    return dA
end


function dmaster_h_heisenberg!(dA, H, J, Jdagger, rates::AbstractVector, A, dA_cache)
    QuantumOpticsBase.mul!(dA,H,A,eltype(A)(im),zero(eltype(A)))
    QuantumOpticsBase.mul!(dA,A,H,-eltype(A)(im),one(eltype(A)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(dA_cache,Jdagger[i],A, eltype(A)(rates[i]),zero(eltype(A)))
        QuantumOpticsBase.mul!(dA,dA_cache,J[i],true,true)

        QuantumOpticsBase.mul!(dA_cache,Jdagger[i],J[i],eltype(A)(-0.5)*rates[i],zero(eltype(A)))
        QuantumOpticsBase.mul!(dA,dA_cache,A,true,true)
        QuantumOpticsBase.mul!(dA,A,dA_cache,true,true)
    end
    return dA
end


function dmaster_h_heisenberg!(dA, H, J, Jdagger, rates::AbstractMatrix, A, dA_cache)
    QuantumOpticsBase.mul!(dA,H,A,eltype(A)(im),zero(eltype(A)))
    QuantumOpticsBase.mul!(dA,A,H,-eltype(A)(im),one(eltype(A)))
    for j=1:length(J), i=1:length(J)
        QuantumOpticsBase.mul!(dA_cache,Jdagger[i],A, eltype(A)(rates[i,j]),zero(eltype(A)))
        QuantumOpticsBase.mul!(dA,dA_cache,J[j],true,true)

        QuantumOpticsBase.mul!(dA_cache,Jdagger[i],J[j],eltype(A)(-0.5)*rates[i,j],zero(eltype(A)))
        QuantumOpticsBase.mul!(dA,dA_cache,A,true,true)
        QuantumOpticsBase.mul!(dA,A,dA_cache,true,true)
    end
    return dA
end
