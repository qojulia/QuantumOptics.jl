function __init__()
    if isdefined(Base.Experimental, :register_error_hint)
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
            if (exc.f == timeevolution.master) && (length(argtypes) >= 3)
                # Check if the given Hamiltonian is constant.
                if !(QuantumOpticsBase.is_const(exc.args[3]))
                    printstyled(io, "\nHint", color=:green)
                    print(io, ": You are attempting to use a time-dependent Hamiltonian with a solver that assumes constant dynamics. To avoid errors, please use the _dynamic solvers instead, e.g. master_dynamic instead of master_dynamic.")
                end
            end

            if (exc.f == timeevolution.schroedinger) && (length(argtypes) >= 3)
                # Check if the given Hamiltonian is constant.
                if !(QuantumOpticsBase.is_const(exc.args[3]))
                    printstyled(io, "\nHint", color=:green)
                    print(io, ": You are attempting to use a time-dependent Hamiltonian with a solver that assumes constant dynamics. To avoid errors, please use the _dynamic solvers instead, e.g. schroedinger_dynamic instead of schroedinger_dynamic.")
                end
            end
        end
    end
end