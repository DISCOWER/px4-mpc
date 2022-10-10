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

## Setup Python Interface
To setup this package on your base Python environemnt, do
```
cd px4-mpc/
poetry build && cd dist && pip install *.whl -U && cd ..
```

To run an example, do
```
python nodes/astrobee_demo.py
```

If you wish to build and run the package in a virtual environment, then do instead
```
poetry install
poetry run astrobee_demo
```


