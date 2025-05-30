using TestItemRunner

@run_package_tests filter=ti->(startswith(ti.filename, "test_") && endswith(ti.filename, ".jl") )