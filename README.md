# Improvements on Detecting Abnormal Deliveries

Team Member 1: Larry Chew

Team Member 2: Kayli Leung

[Presentation](https://docs.google.com/presentation/d/13xDg4bP4t6J1Iv43hvZ-Bq4sVGZcFJ31_uFCNUev_lI/edit?usp=sharing)

## Business Understanding

Deliveries can be stressful for nearly every party involved, including the fetus. Techniques such as the cardiotocogram (CTG) have been helpful and preferable to detecting fetal state. However, at the moment CTGs are interpreted via expert opinion. We want to look into automating the interpretation of the CTG into situations of normal delivery against a suspect or pathological delivery. In a suspect or pathological delivery, the fetus is under distress usually due to hypoxia, acidemia, and/or loss of neural functioning. In these situations, additional steps must be taken to have a healthy delivery. If we can interpret a CTG using a model, we hope to allow delivery teams to be able to put more attention on suspect/pathological cases earlier than waiting for an expert opinion.

## Data Understanding

We used the [UCI Cardiotocography Data Set](https://archive.ics.uci.edu/ml/datasets/cardiotocography#). The dataset has 23 predictive features from CTGs as well as classifications by experts. There were 2126 data points. 

## Data Preparation

We removed columns in the data that were not relevant to our problem. For example, the date was not relevant to this study, nor were the filenames. In addition, there was a second type of classification system for the type of fetal heart rate which we did not use. We OneHotEncoded the categorical feature of `Tendency` and used a standard scalar on our predictors. There were no null data points. We binarized our categories into normal and suspect/pathological as we want to ensure doctors and experts are looking at these cases over the normal cases.

## Modeling

We ran several models on the data to find a model that would minimize the recall. We choose to look at recall because, in most medical situations, it is much more important to have lower false negatives than it is to have low false positives. We grid searched and ran KNN, Decision Trees, Random Forests, and XGBoosting. We settled on XGBoosting with parameters of `learning rate = 0.01`, `n_estimators = 1000`, `max_depth = 8`.

[Notebook](CTG_classification_models.ipynb)

## Evaluation

Our XGBoosted model had a testing recall of about 94% and a Hamming-Loss score of about 3.2%. Our false negative rate was about 6%. We suggest that while this model is helpful and has a fairly low false negative rate, that expert opinion still be used in conjunction with the model. Since this is a life-or-death situation, 6% is a bit too high of a risk to falsely say a birth is normal when it is actually pathological. 

## Future Exploration

We would like to try different threshold values for the predictions to weight towards predicting suspect/pathological over normal. If we had more data we would hopefully be able to improve on our model as we only trained the model on about 1500 data points. If we had images of the CTGs we would like to perform image recognition for classification. This model does not extend to all deliveries as literature has shown CTGs to not be as effective for pre-term/post-term deliveries, multiple births, or for mothers that previously had a C-section.

## Citations:
Ayres de Campos et al. (2000). UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/datasets/cardiotocography#]. Irvine, CA: University of California, School of Information and Computer Science.
