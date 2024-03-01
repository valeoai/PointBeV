zsh scripts/clean.sh
clear
python setup.py build install
python -c 'import torch; import sparse_gs; print(sparse_gs.forward_torch); print(sparse_gs.backward_torch); print(sparse_gs.forward_sparse); print(sparse_gs.backward_sparse); print(sparse_gs.forward_2d_sparse); print(sparse_gs.backward_2d_sparse)'