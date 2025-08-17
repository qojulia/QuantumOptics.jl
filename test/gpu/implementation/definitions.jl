# Test parameters for GPU time evolution tests
const test_sizes = [2, 4, 8]
const max_time_steps = 16
const round_count = 2

# Time evolution parameters
const T_SHORT = [0.0, 0.1, 0.2]
const T_MEDIUM = [0.0:0.1:1.0;]

# Helper function for distance comparison
D(op1::AbstractOperator, op2::AbstractOperator) = abs(tracedistance_nh(dense(op1), dense(op2)))
D(x1::StateVector, x2::StateVector) = norm(x2-x1)

# Test tolerances for GPU computations
const GPU_TOL = 1e-10