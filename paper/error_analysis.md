# Error Analysis

## Objective

The objective of this analysis is to evaluate the prediction errors made by the proposed Adaptive Multi-Layer Perceptron (Adaptive MLP) model on the Chronic Kidney Disease (CKD) dataset.

## False Negative Analysis

False negatives occur when a patient with Chronic Kidney Disease is incorrectly classified as healthy. These errors are particularly critical in healthcare because they may delay diagnosis and treatment.

During testing, the Adaptive MLP achieved a recall score of **1.00**, indicating that no false negative cases were observed. All CKD-positive patients in the test set were correctly identified.

## False Positive Analysis

The model produced **one false positive**, where a healthy patient was incorrectly classified as having CKD. This is reflected in the precision score of **96.77%**.

Although false positives may result in additional clinical examinations, they are generally less critical than false negatives because they do not overlook patients requiring medical treatment.

## Limitations

The evaluation was conducted using a publicly available CKD dataset with a limited number of samples. While the Adaptive MLP achieved excellent predictive performance, additional validation on larger and more diverse clinical datasets is necessary to confirm its robustness and generalization capability. Future work may also investigate the false positive cases to further improve precision while maintaining high recall.