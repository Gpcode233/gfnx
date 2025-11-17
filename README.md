# GFNX: Generative Flow Networks in jaX
GFlowNet Environments in Jax

Installation is handled by `hatch`: https://hatch.pypa.io/latest/

Installtion: 
```
    pip install -e .
```
To install packages for algorithms, use
```
    pip install -e '.[baselines]'
```

TODO: Establish right depenencies everywhere.

## Documentation

To have an access to a local documentation, use the following commands:
```
pip install mkdocs mkdocs-material
pip install "mkdocstrings[python]"
cd gfnx-docs
mkdocs serve
```
Afterwards, you have an access to documentation via localhost.