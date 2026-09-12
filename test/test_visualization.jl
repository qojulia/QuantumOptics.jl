@testitem "Visualization re-exports" begin
    using QuantumOptics
    import QuantumOpticsBase

    for name in (:blochsphereplot, :blochsphereplot!,
                 :fockdistributionplot, :fockdistributionplot!,
                 :wignerplot, :wignerplot!,
                 :wavefunctionplot, :wavefunctionplot!)
        @test name in names(QuantumOptics)
        @test getfield(QuantumOptics, name) === getfield(QuantumOpticsBase, name)
    end
end
