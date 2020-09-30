# Naive Bayes Classifier <br>
* This model predicts the class of a given mushroom based on various features (23 in total) by making use of **Bayes Theorem** and **Naive Bayes Assumption**.

## Bayes Theorem for n features
Here we have a naive bayes assumption and that is that probability of a feature belonging to a class is not affected by the other features.
i.e Let say if a mushroom is blue then it won't give any other information about mushroom.
* Let X have n features and there are total m examples and k classes then ,
* ```P(Y=C/X) = (P(X/Y=C) * P(Y=C))/P(X) = (P(x1.x2.x3.x4...xn / Y=C) * P(Y=C))/P(X)```
* Acc. to bayes theorem ```(P(x1.x2.x3.x4...xn / Y=C) = P(x1/Y=C) * P(x2/Y=C) * P(x3/Y=C) * .....* P(xn/Y=C) ```
* ``` P(X) = P(X/Y=0) * P(Y=0) + P(X/Y=1) * P(Y=1) + .... + P(X/Y=K) * P(Y=K) ```
* So final formula is :-
* ```P(Y=C/X)=(P(x1/Y=C)*P(x2/Y=C)*P(x3/Y=C)* ...*P(xn/Y=C))*P(Y=C) / (P(X/Y=0)*P(Y=0)+P(X/Y=1)*P(Y=1)+...+ P(X/Y=K) * P(Y=K)) ```
* Here P(X) can discarded as we have to find maxm value of posterior prob and P(X) is common in every posterior prob so we can discard it.
