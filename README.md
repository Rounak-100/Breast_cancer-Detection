This repository contains the implementation of a multimodal breast cancer classification framework that integrates ultrasound, histopathological, and chest X-ray MSI images using modality-specific CNN backbones, CBAM-based feature refinement, and an adaptive gated cross-attention fusion mechanism.
Proposed Approach

The proposed framework is designed to effectively model both intra-modality and inter-modality relationships:

Modality-Specific Feature Extraction
Each imaging modality is processed using a dedicated pretrained CNN backbone:

DenseNet121 for Chest X-ray images

EfficientNet-B3 for Histopathological images

ResNet50 for Ultrasound images

A Convolutional Block Attention Module (CBAM) is integrated into each backbone to enhance spatial and channel-wise feature representations by emphasizing diagnostically relevant regions.

Gated Cross-Attention Fusion
Extracted modality-specific features are projected into a shared embedding space and fused using a gated cross-attention mechanism.
The gating strategy dynamically regulates cross-modal information flow, allowing the model to prioritize informative modalities while suppressing noisy or less relevant inputs on a per-sample basis.

Classification
The fused representation is passed to a fully connected layer for binary classification (benign/normal vs. malignant).

A visual overview of the proposed architecture is provided in
proposed_model_architecture.png.
File Descriptions

ResNet-50_and_simple_concatenation.ipynb
Implements the baseline multimodal model using a ResNet-50 backbone and simple feature concatenation without attention mechanisms.

modality_specific_gated_fusion.ipynb
Implements modality-specific CNN backbones with gated fusion, without CBAM, to study the effect of adaptive fusion alone.

proposed_model.ipynb
Implements the complete proposed architecture integrating CBAM-enhanced modality-specific backbones with gated cross-attention fusion.

proposed_model_architecture.png
Diagram illustrating the overall architecture of the proposed framework.
