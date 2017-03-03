using Base.Test
using QuantumOptics


@testset "sortedindices" begin

s = QuantumOptics.sortedindices

@test s.complement(6, [1, 4]) == [2, 3, 5, 6]

@test s.union([1, 4, 5], [2, 4, 7]) == [1, 2, 4, 5, 7]
@test s.union([1, 4, 5], [2, 4]) == [1, 2, 4, 5]
@test s.union([1, 4, 5], [2, 4, 5]) == [1, 2, 4, 5]

@test s.remove([1, 4, 5], [2, 4, 7]) == [1, 5]
@test s.remove([1, 4, 5, 7], [2, 4, 7]) == [1, 5]
@test s.remove([1, 4, 5, 8], [2, 4, 7]) == [1, 5, 8]

@test s.shiftremove([1, 4, 5], [2, 4, 7]) == [1, 3]
@test s.shiftremove([1, 4, 5, 7], [2, 4, 7]) == [1, 3]
@test s.shiftremove([1, 4, 5, 8], [2, 4, 7]) == [1, 3, 5]

@test s.intersect([2, 3, 6], [1, 3, 4, 7]) == [3]
@test s.intersect([2, 3, 6], [1, 3]) == [3]
@test s.intersect([2, 3, 6], [1, 3, 6]) == [3, 6]

@test s.reducedindices([3, 5], [2, 3, 5, 6]) == [2, 3]

x = [3, 5]
s.reducedindices!(x, [2, 3, 5, 6])
@test x == [2, 3]

end # testset
