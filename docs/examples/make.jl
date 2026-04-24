sourcedir = "./notebooks"
markdowndir = "./markdown"
juliadir = "./julia"
targetpath_examples = get(ENV, "TARGETPATH_EXAMPLES", "../QuantumOptics.jl-documentation/src/examples")
jupyter_kernel_name = "julia"

using IJulia
IJulia.installkernel(jupyter_kernel_name, "--project=@."; specname=jupyter_kernel_name, displayname=jupyter_kernel_name)

if !isdir(markdowndir)
    println("Creating markdown output directory at \"", markdowndir, "\"")
    mkdir(markdowndir)
end
if !isdir(juliadir)
    println("Creating julia source output directory at \"", juliadir, "\"")
    mkdir(juliadir)
end

names = filter(name->endswith(name, ".ipynb"), readdir(sourcedir))

ENV["JULIA_PROJECT"] = pwd()

function convert2source(name)
    sourcepath = joinpath(sourcedir, name)
    run(`jupyter nbconvert --to=script --ExecutePreprocessor.kernel_name=$jupyter_kernel_name --ExecutePreprocessor.timeout=200 --output-dir=$juliadir $sourcepath`)
end

function convert2markdown(name)
    sourcepath = joinpath(sourcedir, name)
    run(`jupyter nbconvert --to markdown --ExecutePreprocessor.kernel_name=$jupyter_kernel_name --ExecutePreprocessor.timeout=200 --output-dir=$markdowndir --template=markdown_template.tpl --execute $sourcepath `)
end

overwrite = true

for name in names
    mdname = name[1:end-6] * ".md"
    if overwrite || !isfile(joinpath(markdowndir, mdname))
        # Convert notebook to julia file
        convert2source(name)

        # Execute notebook and convert to rst
        convert2markdown(name)
    end
end

cp(markdowndir, targetpath_examples; force=true)
