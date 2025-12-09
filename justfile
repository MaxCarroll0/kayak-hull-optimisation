
compile:
    @echo "Compiling..."
    @python -m py_compile **/*.py
    @echo "Done"

format:
    @echo "Formatting... (autopep8)"
    @autopep8 --in-place --aggressive --aggressive **/*.py
    @echo "Done"

build: format
    @echo "Building... (TODO)"
    @echo "Done"