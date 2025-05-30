@testitem "Quality Assurance" tags = [:aqua] begin
    using Aqua
    using QuantumOptics

    Aqua.test_all(QuantumOptics, piracies = (broken = true,))

    # manual piracy check to exclude dispatches from QuantumOpticsBase
    phasespace_funcs = [:qfunc, :wigner, :coherentspinstate, :qfuncsu2, :wignersu2]
    pirates = [pirate for pirate in Aqua.Piracy.hunt(QuantumOptics) if pirate.name ∉ phasespace_funcs]
    @test isempty(pirates)
end