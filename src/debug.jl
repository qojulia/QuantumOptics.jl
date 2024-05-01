function __init__()
    if isdefined(Base.Experimental, :register_error_hint)

        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
            if exc.f == timeevolution.master
                printstyled(io, "\nHint", color=:green)
                print(io, ": If your Hamiltonian is time-dependent, then you may want to use master_dynamic instead of master to solve evolution.")
            end
        end

    end
end