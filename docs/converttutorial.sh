#!/bin/bash
jupyter-nbconvert --to=rst --FilesWriter.build_directory=source/tutorial --execute source/tutorial/_tutorial.ipynb