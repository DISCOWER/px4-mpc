# fixedwing-mpc
This package contains an MPC example using the ACADO toolkit

## Setup
Clone and install the [ACADO toolkit](https://acado.github.io/install_linux.html)
```
git clone https://github.com/acado/acado.git -b stable ACADOtoolkit
cd ACADOtoolkit
mkdir build
cd build
cmake ..
make
```

Build this package
```
mkdir build
cd build
cmake ..
make
```

## Setup Python Interface - System Install

This package uses the Poetry package manager to manage dependencies. To install poetry, follow the instructions [here](https://python-poetry.org/docs/#installation). Once poetry is installed, setup this package on your base Python environment with
```
cd px4-mpc/
poetry build && cd dist && pip install *.whl -U && cd ..
```

To run an example, do
```
python nodes/astrobee_demo.py
python nodes/quadcopter_demo.py
```

## Setup Python Interface - Poetry .venv Install
If you wish to build and run the package in a Poetry virtual environment, then do instead
```
cd px4-mpc/
poetry install
```

Then, to run an example, do
```
poetry run astrobee_demo
poetry run quadcopter_demo
```

