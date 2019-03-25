using Test
using QuantumOptics


@testset "sortedindices" begin

s = QuantumOptics.sortedindices

@test s.complement(6, [1, 4]) == [2, 3, 5, 6]

@test s.remove([1, 4, 5], [2, 4, 7]) == [1, 5]
@test s.remove([1, 4, 5, 7], [2, 4, 7]) == [1, 5]
@test s.remove([1, 4, 5, 8], [2, 4, 7]) == [1, 5, 8]

@test s.shiftremove([1, 4, 5], [2, 4, 7]) == [1, 3]
@test s.shiftremove([1, 4, 5, 7], [2, 4, 7]) == [1, 3]
@test s.shiftremove([1, 4, 5, 8], [2, 4, 7]) == [1, 3, 5]

@test s.reducedindices([3, 5], [2, 3, 5, 6]) == [2, 3]
x = [3, 5]
s.reducedindices!(x, [2, 3, 5, 6])
@test x == [2, 3]

@test_throws AssertionError s.check_indices(5, [1, 6])
@test_throws AssertionError s.check_indices(5, [0, 2])
@test s.check_indices(5, Int[]) == nothing
@test s.check_indices(5, [1, 3]) == nothing
@test s.check_indices(5, [3, 1]) == nothing

@test_throws AssertionError s.check_sortedindices(5, [1, 6])
@test_throws AssertionError s.check_sortedindices(5, [3, 1])
@test_throws AssertionError s.check_sortedindices(5, [0, 2])
@test s.check_sortedindices(5, Int[]) == nothing
@test s.check_sortedindices(5, [1, 3]) == nothing

end # testset
