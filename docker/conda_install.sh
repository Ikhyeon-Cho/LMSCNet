
#!/bin/bash

# Install additional packages in the active conda environment
conda install -n $1 \
    numpy \
    tqdm \
    scipy \
    scikit-learn \
    pyyaml \
    imageio \
    tensorboard \
    -y