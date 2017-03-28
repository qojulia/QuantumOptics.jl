import os
import subprocess

sourcedir = "../examples"
builddir = "source/examples"

names = os.listdir(sourcedir)
names = filter(lambda x: x.endswith(".ipynb"), names)
# names = filter(lambda x: "raman" in x, names)

for name in names:
    barename, extension = os.path.splitext(name)
    sourcepath = os.path.join(sourcedir, name)
    targetpath = os.path.join(builddir, barename + ".rst")

    # Convert notebook to julia file
    subprocess.run([
        "jupyter-nbconvert",
                "--to=script",
                "--FileWriter.build_directory=" + sourcedir,
                sourcepath])

    # Execute notebook and convert to rst
    try:
        subprocess.run([
            "jupyter-nbconvert",
                    "--to", "rst",
                    "--output-dir", builddir,
                    "--execute",
                    sourcepath],
            check=True)

        # Set restructered text link label
        f = open(targetpath)
        text = f.read()
        f.close()
        pos = text.find("\n==")
        pos = text.rfind("\n", 0, pos)
        text = text[:pos] + "\n.. _example-{}:\n".format(barename) + text[pos:]
        f = open(targetpath, "w")
        f.write(text)
        f.close()
    except:
        print("Could not convert ", name)
