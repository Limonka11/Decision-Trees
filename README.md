# Intro to Machine Learning CW1

To connect to a lab machine, first ssh into a login server (for example):
### ssh [username]@shell2.doc.ic.ac.uk

Then you can connect to a lab machine using (for example):
### ssh texel19

The repository can then be cloned or copied to the machine and the relevant packages installed with the command:
### export PYTHONUSERBASE=/vol/lab/ml/mlenv

To run the code, use the command
### python3 main.py

The code will pull the clean and noisy datasets from 'wifi_db/clean_dataset.txt' and 'wifi_db/noisy_dataset.txt' respectively.

This will first produce images of non-pruned and then pruned trees trained on
a single instance of a training, validation and test set. It will also print
some evaluation results to standard output.

The repo comes with pre-produced images of such trees but running the code
will produce another set and save them in the files:
#### clean_set_no_pruning.png
#### clean_set_with_pruning.png
#### noisy_set_no_pruning.png
#### noisy_set_with_pruning.png

After that, a 10-fold cross evaluation will be performed on both the clean and
noisy datasets. Results and evaluation will be printed to standard output.

This last part takes a bit to run. Each run will produce different results
since each time the datasets are randomly shuffled.

Necessary libraries to be installed are numpy and matplotlib.
