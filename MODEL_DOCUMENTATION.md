# Otomoto ANN Model Documentation

**Rowland M Bernard**
**January 2026**

## Overview

This document provides comprehensive technical documentation for the Artificial Neural Network (ANN) models developed for Otomoto's customer churn prediction and marketing segmentation system.

---

## Model Architectures

### 1. Baseline Model (Unoptimized)

**Purpose**: Establish performance baseline for comparison with optimized versions

![Baseline Model Architecture](diagrams/baseline_model_architecture.drawio%20(1).png)

**Architecture Details**:
```
Input Layer: n_features (40 neurons)
    ↓
Hidden Layer 1: Dense(64, ReLU)
    ↓
Hidden Layer 2: Dense(32, ReLU)
    ↓
Hidden Layer 3: Dense(16, ReLU)
    ↓
Output Layer: Dense(1, Sigmoid)
```

**Configuration Parameters**:
- **Optimizer**: Vanilla SGD (learning_rate=0.01)
- **Loss Function**: Binary Cross-Entropy
- **Batch Size**: 32
- **Epochs**: 50
- **Input Dimension**: 40 features
- **Total Parameters**: 3,169

**Layer-wise Parameter Breakdown**:
| Layer | Input Size | Output Size | Parameters | Activation |
|-------|------------|-------------|------------|------------|
| Input | 40 | 64 | 2,640 | ReLU |
| Hidden 1 | 64 | 32 | 2,080 | ReLU |
| Hidden 2 | 32 | 16 | 528 | ReLU |
| Output | 16 | 1 | 17 | Sigmoid |

**Strengths**:
- Simple, interpretable architecture
- Low computational overhead
- Sufficient capacity for binary classification
- Appropriate output activation (Sigmoid)

**Identified Weaknesses**:
- No regularization mechanisms
- Basic optimizer without momentum
- Fixed learning rate
- Susceptible to overfitting
- Potential vanishing gradient issues

---

### 2. Optimized Model Architecture

**Purpose**: Enhanced performance through regularization and advanced training techniques

![Optimized Model Architecture](diagrams/optimized_model_architecture.drawio.png)

**Architecture Details**:
```
Input Layer: n_features (40 neurons)
    ↓
Dense Layer: Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense Layer: Dense(64, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense Layer: Dense(32, ReLU) → BatchNorm → Dropout(0.2)
    ↓
Dense Layer: Dense(16, ReLU)
    ↓
Output Layer: Dense(1, Sigmoid)
```

**Configuration Parameters**:
- **Batch Size**: 32
- **Maximum Epochs**: 100 (with early stopping)
- **Validation Split**: 0.2
- **Input Dimension**: 40 features
- **Regularization**: Batch Normalization + Dropout
- **Early Stopping**: Patience=15, monitor='val_loss'
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)

**Layer-wise Parameter Breakdown**:
| Layer | Input Size | Output Size | Parameters | Regularization | Activation |
|-------|------------|-------------|------------|----------------|------------|
| Input | 40 | 128 | 5,248 | BatchNorm, Dropout(0.3) | ReLU |
| Hidden 1 | 128 | 64 | 8,256 | BatchNorm, Dropout(0.3) | ReLU |
| Hidden 2 | 64 | 32 | 2,080 | BatchNorm, Dropout(0.2) | ReLU |
| Hidden 3 | 32 | 16 | 528 | None | ReLU |
| Output | 16 | 1 | 17 | None | Sigmoid |

**Total Parameters**: 16,129

![Optimization Algorithm Comparison](diagrams/optimization_algorithm_comparison.drawio%20(1).png)

---

## Optimization Algorithms

### 1. Adam Optimizer

**Mathematical Foundation**:
```
m_t = β₁ * m_(t-1) + (1 - β₁) * g_t
v_t = β₂ * v_(t-1) + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_t = θ_(t-1) - α * m̂_t / (√v̂_t + ε)
```

**Hyperparameters**:
- **Learning Rate (α)**: 0.001
- **β₁ (Momentum)**: 0.9
- **β₂ (RMSprop)**: 0.999
- **ε (Numerical Stability)**: 1e-7

**Advantages for Otomoto**:
- Adaptive learning rates handle feature scale differences
- Robust to noisy marketing data
- Fast convergence with minimal hyperparameter tuning
- Effective with sparse one-hot encoded features

### 2. RMSprop Optimizer

**Mathematical Foundation**:
```
E[g²]_t = ρ * E[g²]_(t-1) + (1 - ρ) * g_t²
θ_t = θ_(t-1) - α * g_t / √(E[g²]_t + ε)
```

**Hyperparameters**:
- **Learning Rate (α)**: 0.001
- **Decay Rate (ρ)**: 0.9
- **ε**: 1e-7

**Advantages for Otomoto**:
- Handles non-stationary customer behavior patterns
- Effective for evolving market conditions
- Lower memory requirements than Adam
- Good performance on complex loss landscapes

### 3. SGD with Momentum and Nesterov Acceleration

**Mathematical Foundation**:
```
v_t = μ * v_(t-1) - α * ∇f(θ_(t-1) + μ * v_(t-1))
θ_t = θ_(t-1) + v_t
```

**Hyperparameters**:
- **Learning Rate (α)**: 0.01
- **Momentum (μ)**: 0.9
- **Nesterov**: True

**Advantages for Otomoto**:
- Classical approach with proven theoretical properties
- Better generalization capabilities
- Effective for escaping local minima
- Reduced oscillations during training

### 4. Adagrad Optimizer

**Mathematical Foundation**:
```
G_t = G_(t-1) + g_t²
θ_t = θ_(t-1) - α * g_t / √(G_t + ε)
```

**Hyperparameters**:
- **Learning Rate (α)**: 0.01
- **ε**: 1e-7

**Advantages for Otomoto**:
- Excellent for sparse categorical features
- Feature-specific learning rate adaptation
- No manual learning rate tuning required
- Strong performance on infrequent feature patterns

---

## Data Preprocessing Pipeline

![Data Processing Pipeline](diagrams/data_processing_pipeline.drawio.png)

### Feature Engineering Process

**1. Data Cleaning**:
- **Customer ID Removal**: Non-predictive identifier
- **TotalCharges Conversion**: String to numeric with median imputation
- **Missing Value Handling**: <0.2% missing values imputed using median

**2. Categorical Encoding**:
- **Binary Features**: Label Encoding (6 variables)
  - gender, Partner, Dependents, PhoneService, PaperlessBilling, Churn
- **Multi-class Features**: One-Hot Encoding (10 variables)
  - MultipleLines, InternetService, OnlineSecurity, OnlineBackup
  - DeviceProtection, TechSupport, StreamingTV, StreamingMovies
  - Contract, PaymentMethod

**3. Feature Scaling**:
- **Method**: StandardScaler (Z-score normalization)
- **Purpose**: Zero mean, unit variance
- **Impact**: Improves convergence speed and stability

**Final Feature Set**:
- **Total Features**: 40
- **Numerical Features**: 3 (tenure, MonthlyCharges, TotalCharges)
- **Binary Features**: 6
- **One-Hot Encoded Features**: 31

**Data Splitting Strategy**:
- **Training Set**: 80% (5,634 records)
- **Test Set**: 20% (1,409 records)
- **Stratification**: Maintains 26.5% churn rate across splits
- **Validation**: 20% of training data for early stopping

---

## Model Training Configuration

### Training Parameters

**All Optimized Models**:
```python
training_config = {
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "verbose": 1,
    "callbacks": [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
}
```

**Early Stopping Strategy**:
- **Monitor**: Validation loss
- **Patience**: 15 epochs
- **Restore Best**: Weights from best epoch
- **Purpose**: Prevent overfitting and optimize training time

**Learning Rate Scheduling**:
- **Strategy**: ReduceLROnPlateau
- **Reduction Factor**: 0.5
- **Trigger**: No improvement in validation loss for 5 epochs
- **Minimum Learning Rate**: 1e-7

---

## Evaluation Metrics

### Performance Measurement Framework

**1. Classification Metrics**:
```python
metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred),
    "recall": recall_score(y_true, y_pred),
    "f1_score": f1_score(y_true, y_pred),
    "roc_auc": roc_auc_score(y_true, y_pred_proba),
    "loss": model.evaluate(X_test, y_test, verbose=0)[0]
}
```

**2. Business Context Interpretation**:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Marketing budget efficiency (minimize false positives)
- **Recall**: Customer retention effectiveness (minimize false negatives)
- **F1-Score**: Balanced performance for marketing decisions
- **ROC-AUC**: Model discriminative ability
- **Loss**: Training optimization quality

**3. Confusion Matrix Analysis**:
```
                Predicted
               No Churn   Churn
Actual No Churn    TN      FP
Actual Churn        FN      TP
```

**Business Impact**:
- **False Negatives (FN)**: Lost revenue opportunity (high cost)
- **False Positives (FP)**: Wasted marketing spend (lower cost)
- **Optimal Balance**: Maximize recall while maintaining acceptable precision

---

## Model Performance Analysis

### Expected Performance Characteristics

**Baseline Model Expectations**:
- **Accuracy**: 73-78%
- **F1-Score**: 0.55-0.65
- **ROC-AUC**: 0.70-0.75
- **Training Issues**: Slow convergence, high variance

**Optimized Model Improvements**:
- **Accuracy Gain**: +3-8 percentage points
- **F1-Score Gain**: +5-12 percentage points
- **ROC-AUC Gain**: +0.03-0.08 points
- **Training Speed**: 2-3x faster convergence

### Model Selection Criteria

**Primary Metric**: F1-Score (balance of precision and recall)
**Secondary Metrics**: ROC-AUC, Training Stability
**Business Considerations**: Computational Efficiency, Interpretability

---

## Business Impact and Integration Strategy

![Business Impact & Segmentation Strategy](diagrams/business_impact_strategy.drawio%20(1).png)

---

## Deployment Architecture

### Model Serving Requirements

**Input Processing Pipeline**:
```python
def preprocess_input(customer_data):
    # 1. Feature engineering
    # 2. Categorical encoding
    # 3. Feature scaling
    # 4. Shape adjustment
    return processed_data
```

**Prediction Service**:
```python
def predict_churn_probability(customer_data):
    processed_data = preprocess_input(customer_data)
    churn_probability = model.predict(processed_data)
    return churn_probability[0][0]
```

**Batch Processing Capabilities**:
- **Throughput**: 1000+ predictions/second
- **Latency**: <100ms per prediction
- **Memory Footprint**: <500MB model size

---

## Monitoring and Maintenance

### Performance Tracking

**Daily Metrics**:
- Prediction volume and accuracy
- Feature distribution monitoring
- Model drift detection

**Weekly Analysis**:
- Business impact assessment
- A/B test results
- Customer satisfaction metrics

**Monthly Reviews**:
- Model retraining decisions
- Feature engineering updates
- Performance baseline adjustments

### Model Update Strategy

**Triggers for Retraining**:
- Performance degradation >5%
- Feature distribution drift >10%
- New customer segments identified
- Major business process changes

**Retraining Process**:
1. Collect new labeled data
2. Validate data quality
3. Train and validate new model
4. A/B test against production model
5. Gradual rollout if improvements confirmed

---

## Technical Specifications

### Hardware Requirements

**Training Environment**:
- **CPU**: 4+ cores (recommended: 8+ cores)
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: Optional but recommended for faster training
- **Storage**: 5GB for data and models

**Production Environment**:
- **CPU**: 2+ cores
- **RAM**: 4GB minimum
- **GPU**: Not required for inference
- **API**: RESTful endpoint for predictions

### Software Dependencies

**Core Libraries**:
```python
tensorflow==2.12.0
scikit-learn==1.2.0
pandas==1.5.3
numpy==1.23.5
matplotlib==3.7.1
seaborn==0.12.2
```

**Development Tools**:
- **Jupyter Notebook**: Model development and experimentation
- **Git**: Version control
- **Docker**: Containerization for deployment
- **Kubernetes**: Orchestration (optional)

---

## Security and Compliance

### Data Privacy Considerations

**Customer Data Protection**:
- No PII stored in model weights
- Encrypted data transmission
- Access logging and audit trails
- GDPR compliance for EU customers

**Model Security**:
- Input validation and sanitization
- Rate limiting for API endpoints
- Model integrity verification
- Secure model storage and versioning

### Business Compliance

**Regulatory Requirements**:
- Fair lending laws compliance
- Non-discriminatory model behavior
- Transparent decision-making processes
- Regular bias audits

---

## Scalability Planning

### Horizontal Scaling

**Load Balancing**:
- Multiple model instances behind load balancer
- Auto-scaling based on prediction demand
- Geographic distribution for global operations

**Caching Strategy**:
- Redis for frequent predictions
- Model weight caching in memory
- Feature preprocessing result caching

### Vertical Scaling

**Performance Optimization**:
- GPU acceleration for batch predictions
- Model quantization for edge deployment
- Pruning and compression techniques

---

## Integration Points

### CRM Integration

**Data Flow**:
1. Customer data → CRM system
2. Feature extraction → Preprocessing pipeline
3. Model prediction → Risk scores
4. Risk scores → Marketing campaigns

**API Specifications**:
```
POST /api/predict/churn
{
    "customer_id": "CUST123456",
    "features": {...}
}
Response:
{
    "churn_probability": 0.73,
    "risk_level": "high",
    "recommendation": "retention_campaign"
}
```

### Marketing Automation

**Campaign Trigger Logic**:
```python
if churn_probability > 0.7:
    trigger_high_risk_campaign(customer_id)
elif churn_probability > 0.4:
    trigger_medium_risk_campaign(customer_id)
else:
    trigger_upsell_campaign(customer_id)
```

---

## Troubleshooting Guide

### Common Issues

**Training Problems**:
1. **Vanishing Gradients**: Use batch normalization, proper initialization
2. **Overfitting**: Increase dropout, reduce model complexity, add more data
3. **Poor Convergence**: Adjust learning rate, change optimizer
4. **Memory Issues**: Reduce batch size, use gradient accumulation

**Prediction Issues**:
1. **Input Shape Mismatch**: Verify preprocessing pipeline consistency
2. **Feature Distribution Drift**: Monitor feature statistics, retrain model
3. **Performance Degradation**: Schedule regular model updates

### Debugging Tools

**Model Inspection**:
```python
model.summary()
model.get_weights()
model.predict_proba(sample_input)
```

**Performance Profiling**:
```python
import time
start_time = time.time()
predictions = model.predict(batch_data)
inference_time = time.time() - start_time
```

---

## Best Practices

### Development Workflow

1. **Data Validation**: Always validate input data quality
2. **Version Control**: Track model versions, hyperparameters, and data
3. **Testing**: Comprehensive unit and integration tests
4. **Documentation**: Maintain detailed model cards and documentation
5. **Review Process**: Code and model review before deployment

### Model Lifecycle Management

1. **Experiment Tracking**: Log all experiments with metrics
2. **Model Registry**: Centralized model version management
3. **Automated Testing**: Continuous integration for model validation
4. **Rollback Strategy**: Quick reversion capability for failed deployments
5. **Performance Monitoring**: Real-time model performance tracking

---

## Future Enhancements

### Advanced Architectures

1. **Attention Mechanisms**: Improve feature importance understanding
2. **Residual Connections**: Deeper networks without degradation
3. **Transformer Networks**: Sequence-based customer behavior modeling
4. **Graph Neural Networks**: Customer relationship modeling

### Optimization Techniques

1. **Hyperparameter Optimization**: Bayesian optimization, grid search
2. **Neural Architecture Search**: Automated model design
3. **Knowledge Distillation**: Model compression while preserving performance
4. **Federated Learning**: Privacy-preserving distributed training

### Business Intelligence

1. **Customer Lifetime Value Prediction**: Integrated with churn prediction
2. **Next Best Action**: Real-time campaign optimization
3. **Market Basket Analysis**: Product recommendation integration
4. **Competitive Intelligence**: Market trend analysis integration

---

## Conclusion

This documentation provides a comprehensive reference for the Otomoto ANN models, covering architecture details, optimization algorithms, deployment strategies, and maintenance procedures. The models are designed to balance performance, interpretability, and operational efficiency for marketing segmentation applications.

The modular architecture allows for continuous improvement and adaptation to changing business requirements while maintaining robust performance in customer churn prediction tasks.

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintained By**: Otomoto ML Engineering Team