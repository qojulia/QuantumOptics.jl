using IJulia

examples_dir = @__DIR__
notebooks_dir = joinpath(examples_dir, "notebooks")
markdown_dir = joinpath(examples_dir, "markdown")
julia_dir = joinpath(examples_dir, "julia")
docs_examples_dir = normpath(joinpath(examples_dir, "..", "src", "examples"))
template_path = joinpath(examples_dir, "markdown_template.tpl")
kernel_name = "quantumoptics-docs"
timeout = 200
python = get(ENV, "PYTHON", "python3")

IJulia.installkernel(
    kernel_name,
    "--project=$(examples_dir)";
    specname = kernel_name,
    displayname = kernel_name,
)

for dir in (markdown_dir, julia_dir, docs_examples_dir)
    rm(dir; recursive = true, force = true)
    mkpath(dir)
end

notebooks = sort(filter(endswith(".ipynb"), readdir(notebooks_dir; join = true)))

for notebook in notebooks
    run(`$python -m nbconvert --to script --output-dir=$julia_dir $notebook`)
    run(`$python -m nbconvert --to markdown --execute
        --ExecutePreprocessor.kernel_name=$kernel_name
        --ExecutePreprocessor.timeout=$timeout
        --output-dir=$markdown_dir
        --template-file=$template_path
        $notebook`)
end

cp(markdown_dir, docs_examples_dir; force = true)
