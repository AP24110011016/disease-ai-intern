# Cross-Dataset Generalization

## Objective

The objective of this experiment was to evaluate whether the proposed Adaptive Multi-Layer Perceptron (Adaptive MLP) can generalize to a different medical dataset.

## Experimental Setup

The Adaptive MLP was first developed and evaluated using the Pima Indians Diabetes dataset. To verify its robustness, the same preprocessing pipeline and adaptive training strategy were applied to the UCI Chronic Kidney Disease (CKD) dataset.

The preprocessing steps included:

- Missing value handling
- Feature normalization using Min-Max Scaling
- Train-test split (80:20)
- Adaptive neural network training

## Results

| Dataset | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------|----------|-----------|--------|----------|---------|
| Pima Diabetes | 97.50% | 93.75% | 100% | 96.77% | 99.90% |
| CKD | 100% | 100% | 100% | 100% | 100% |

## Discussion

The Adaptive MLP achieved excellent predictive performance on both datasets. Despite differences in disease characteristics and feature distributions, the model maintained high accuracy and F1-score. This demonstrates that the proposed adaptive learning framework is capable of generalizing across multiple healthcare datasets rather than being limited to a single disease.

## Conclusion

The cross-dataset evaluation confirms that the Adaptive MLP is robust and generalizable. These findings strengthen the practical applicability of the proposed framework for disease prediction across different medical domains.