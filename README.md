# alignment_adjust
This project aims to propose a unsupervised learning method to adjust boundaries(forced-alignment results) generated by any forced-alignment tools(such as Kaldi).

## Requirements
* python3
* pandas
* scipy
* bunch
* tensorflow 1.4+

## Algorithm Framework(Expectation Maximization Algorithm)
* E-step
    1. For each boundary, pick up three groups of features, consisting of series of sampling points.
    2. Label these features with 0(too left), 1(just in time) or 2(too right).
    3. Train a classifier with these feature-label.
* M-step
    4. Predict label of features using the trained classifier.
    5. Adjust each boundary with the predicted labels of its three groups of features.

## Details
* How to pick up three groups of features for a boundary?
* How to adjust each boundary with the predicted labels?
