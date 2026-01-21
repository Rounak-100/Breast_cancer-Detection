# Multimodal Breast Cancer Classification
## Proposed Approach
The proposed framework performs multimodal breast cancer classification by integrating ultrasound, histopathological, and chest X-ray MSI images. Each modality is processed using a dedicated pretrained CNN backbone augmented with a Convolutional Block Attention Module (CBAM) to enhance spatial and channel-wise feature representations. The extracted modality-specific features are projected into a shared embedding space and fused using a gated cross-attention mechanism, which dynamically regulates inter-modal information flow. The fused representation is used for binary classification of benign/normal and malignant cases.
## Files

- **ResNet-50_and_simple_concatenation.ipynb**  
  Baseline multimodal model using ResNet-50 backbones with simple feature concatenation (no attention).

- **modality_specific_gated_fusion.ipynb**  
  Modality-specific CNN backbones with gated fusion (without CBAM).

- **proposed_model.ipynb**  
  Complete proposed model integrating CBAM-enhanced modality-specific backbones with gated cross-attention fusion.

- **proposed_model_architecture.png**  
  Visual illustration of the proposed architecture.
## First, Run

```bash
pip install -r requirements.txt
```
## Dataset
The experiments are conducted using the Breast Cancer MSI Multimodal Image Dataset.

The dataset is publicly available at:
```bash
 https://tinyurl.com/3c44m8ws
```
Update the dataset path inside each notebook before execution.

## Train
### ResNet-50 + Simple Concatenation
```bash
Open and run:
ResNet-50_and_simple_concatenation.ipynb
```
### Modality-Specific Gated Fusion Model
```bash
Open and run:
modality_specific_gated_fusion.ipynb
```
### Proposed CBAM + Gated Cross-Attention Model
```bash
Open and run:
proposed_model.ipynb
```
