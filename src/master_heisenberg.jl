


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


function dmaster_h_heisenberg!(drho, H, J, Jdagger, rates::Nothing, rho, drho_cache)
    QuantumOpticsBase.mul!(drho,H,rho,eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,H,-eltype(rho)(im),one(eltype(rho)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(drho_cache,Jdagger[i],rho)
        QuantumOpticsBase.mul!(drho,drho_cache,J[i],true,true)

        QuantumOpticsBase.mul!(drho,Jdagger[i],drho_cache,eltype(rho)(-0.5),one(eltype(rho)))

        QuantumOpticsBase.mul!(drho_cache,rho,Jdagger[i],true,false)
        QuantumOpticsBase.mul!(drho,drho_cache,J[i],eltype(rho)(-0.5),one(eltype(rho)))
    end
    return drho
end

function dmaster_h_heisenberg!(drho, H, J, Jdagger, rates::AbstractVector, rho, drho_cache)
    QuantumOpticsBase.mul!(drho,H,rho,eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,H,-eltype(rho)(im),one(eltype(rho)))
    for i=1:length(J)
        QuantumOpticsBase.mul!(drho_cache,Jdagger[i],rho,eltype(rho)(rates[i]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,drho_cache,J[i],true,true)

        QuantumOpticsBase.mul!(drho,Jdagger[i],drho_cache,eltype(rho)(-0.5),one(eltype(rho)))

        QuantumOpticsBase.mul!(drho_cache,rho,Jdagger[i],eltype(rho)(rates[i]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,drho_cache,J[i],eltype(rho)(-0.5),one(eltype(rho)))
    end
    return drho
end


function dmaster_h_heisenberg!(drho, H, J, Jdagger, rates::AbstractMatrix, rho, drho_cache)
    QuantumOpticsBase.mul!(drho,H,rho,eltype(rho)(im),zero(eltype(rho)))
    QuantumOpticsBase.mul!(drho,rho,H,-eltype(rho)(im),one(eltype(rho)))
    for j=1:length(J), i=1:length(J)
        QuantumOpticsBase.mul!(drho_cache,Jdagger[i],rho,eltype(rho)(rates[i,j]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,drho_cache,J[j],true,true)

        QuantumOpticsBase.mul!(drho,Jdagger[j],drho_cache,eltype(rho)(-0.5),one(eltype(rho)))

        QuantumOpticsBase.mul!(drho_cache,rho,Jdagger[j],eltype(rho)(rates[i,j]),zero(eltype(rho)))
        QuantumOpticsBase.mul!(drho,drho_cache,J[i],eltype(rho)(-0.5),one(eltype(rho)))
    end
    return drho
end
