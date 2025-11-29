# Build Instructions
Ensure Python3, pip, and Just are installed. Use pip to install:
- autopep8
- PyFoam
- numpy
- matplotlib

Build the project with `just build`.

## Nix
For reproducable builds. Using nix, set up dependencies using the following [flake](https://github.com/MaxCarroll0/kayak-hull-optimisation-flake) as a nix-shell / direnv.

Build the project with `just build`.
