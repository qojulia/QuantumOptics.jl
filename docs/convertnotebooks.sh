#!/bin/bash
SOURCEPATH=../examples/
BUILDPATH=source/examples/

for f in $SOURCEPATH*.ipynb
do
    jupyter nbconvert --to=rst --FilesWriter.build_directory=$BUILDPATH "$f"
    jupyter nbconvert --to script --FilesWriter.build_directory=$SOURCEPATH "$f"
done
