                
# Optimal Classification under Performative Distribution Shift

This repository contains the Python implementation of the paper Optimal Classification under Performative Distribution Shift.

Please cite this paper if you use our code.

```
@inproceedings{cyffers2024performative,
  author = {Edwige Cyffers, Muni Sreenivas Pydi, Jamal Atif, Olivier Capp{\'e}},
  title = {Optimal Classification under Performative Distribution Shift },
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  volume  = {37}
}
```

## Reproducing papers figures

For figure 1, use the jupyter notebook.

For figure 2 (a) and (b), run synthlog.py

For figure 2 (c), run quadRRM.py

For figure 2 (d) and (e), run withlearning.py

For figure 2 (f) run creditdatasetpreprocess.py then comphouses.py then coloredhouses.py. The credit dataset can be downloaded from www.kaggle.com/c/GiveMeSomeCredit/data

## some help for understanding the code

Main code is in learner.py where the different algorithms are implemented. The implementation for learning Pi is limited to diagonal matrices. Running the SF estimator produce warning due to overflow during execution, showcasing that our estimator is more stable than this baseline.
