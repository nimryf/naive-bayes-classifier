# A binary Naive-Bayes classifier.

The classifier distinguishes two classes: 0 for ham emails and 1 for spam emails.

- It has no external dependencies and only uses two internal libraries.
- Training and testing sets have been provided.
- Laplace smoothing is used when estimating class priors.
- The numerical stability of the algorithm is increased by taking logarithms of posterior distributions.
- Both class priors and conditional likelihoods are estimated from training data only.

The training dataset consists of 1000 rows and 55 columns:

- Each row corresponds to one email message.
- The first column is the response variable, i.e. 0 or 1.
- The other 54 columns are features that correspond to 54 different keywords and special characters.
