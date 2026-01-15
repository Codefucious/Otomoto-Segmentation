# Otomoto Project Diagrams

This folder contains draw.io XML diagrams for the Otomoto Marketing Segmentation Optimization project.

## Diagram Files

### 1. Baseline Model Architecture
**File:** `baseline_model_architecture.xml`

Visual representation of the unoptimized baseline ANN model:
- Input layer with 40 features
- Three hidden layers (64, 32, 16 neurons)
- Output layer with sigmoid activation
- Configuration parameters
- Model characteristics

### 2. Optimized Model Architecture  
**File:** `optimized_model_architecture.xml`

Enhanced model architecture with regularization:
- Deeper network (128, 64, 32, 16 neurons)
- Batch normalization layers
- Dropout regularization (0.3, 0.3, 0.2)
- Early stopping and learning rate scheduling
- Improved configuration parameters

### 3. Optimization Algorithm Comparison
**File:** `optimization_algorithm_comparison.xml`

Comprehensive comparison of all optimization algorithms:
- Baseline SGD characteristics and weaknesses
- Adam optimizer advantages and hyperparameters
- RMSprop for non-stationary objectives
- SGD with Momentum benefits
- Adagrad for sparse features
- Performance comparison matrix
- Expected performance characteristics

### 4. Data Processing Pipeline
**File:** `data_processing_pipeline.xml`

End-to-end data workflow visualization:
- Raw data ingestion and cleaning
- Feature engineering pipeline
- Binary, categorical, and numerical feature processing
- Train-test split strategy
- Model training and evaluation phases
- Quality assurance processes

### 5. Business Impact & Segmentation Strategy
**File:** `business_impact_strategy.xml`

Business-focused visual representation:
- Customer segmentation (high, medium, low risk)
- Targeted marketing campaigns
- Expected business impact and ROI
- Success metrics and monitoring
- Implementation timeline
- Cost-benefit analysis

## Usage Instructions

### Opening the Diagrams
1. Open draw.io (https://app.diagrams.net/)
2. Click "Open Existing Diagram"
3. Select "Open from your device"
4. Navigate to the XML file location
5. The diagram will render in the draw.io interface

### Customization Options
- **Colors:** Modify fill colors to match corporate branding
- **Fonts:** Adjust font sizes and styles for presentations
- **Layout:** Rearrange elements to focus on specific aspects
- **Export:** Save as PNG, SVG, PDF for different use cases

### Integration with Documentation
These diagrams are referenced throughout:
- **Technical Report:** Section 3.2 Model Architecture
- **Model Documentation:** Architecture overview
- **Presentations:** Executive summary visuals
- **Documentation:** Training materials for stakeholders

## Technical Specifications

### Draw.io XML Format
- **Standard:** mxGraph XML format
- **Compatibility:** draw.io desktop and web versions
- **Version:** Compatible with draw.io v21.6.5+
- **Encoding:** UTF-8

### Design Principles
- **Color Coding:** Consistent use of colors for different components
- **Information Hierarchy:** Clear visual hierarchy with sizing and positioning
- **Clarity:** Minimal text with maximum information density
- **Professional:** Business-appropriate styling and corporate colors

## Diagram Components Legend

### Color Meanings
- 🔵 **Blue:** Input layers, data processing
- 🟢 **Green:** Positive outcomes, optimized components
- 🟠 **Orange:** Warning states, attention areas
- 🔴 **Red:** High risk, baseline limitations
- 🟣 **Purple:** Classical methods, proven approaches

### Shape Meanings
- **Rounded Rectangles:** Main components
- **Rectangles:** Containers and groupings
- **Arrows:** Data/process flow
- **Text Blocks:** Descriptions and specifications

## Updates and Maintenance

### Version Control
- Each diagram is version-controlled with date stamps
- Changes should be tracked in commit messages
- Maintain backward compatibility when possible

### Updating Process
1. Open the XML file in draw.io
2. Make necessary modifications
3. Export updated XML file
4. Update corresponding documentation references
5. Test visualization in different formats

## Troubleshooting

### Common Issues
- **XML Not Loading:** Ensure proper XML formatting with correct header
- **Missing Elements:** Check for corrupted XML tags
- **Styling Issues:** Verify color codes and font specifications
- **Layout Problems:** Reset to default layout and reposition elements

### Support
For diagram-related issues:
- Check draw.io documentation for XML format requirements
- Validate XML syntax using online validators
- Test with different draw.io versions if compatibility issues arise

---

**Created By:** Otomoto ML Engineering Team  
**Date:** January 2026  
**Version:** 1.0  
**Format:** Draw.io XML