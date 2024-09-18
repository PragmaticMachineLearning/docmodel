# DocModel: A Document Understanding Model

DocModel is a state-of-the-art document understanding model designed to extract both textual content and 2D spatial relationships from complex documents. Built on the RoBERTa architecture, DocModel is fine-tuned for tasks such as form understanding, entity extraction, and layout-based document analysis.

### Key Features

2D Spatial Modeling: Captures text layout and spatial relationships within documents, ideal for complex document structures such as forms, tables, and scans.
RoBERTa-based Architecture: Built on a robust architecture for token-level tasks with powerful self-attention mechanisms.
Fine-tuned for Document Understanding: Specifically trained on datasets like FUNSD to handle noisy and complex document layouts.

### Model Performance

DocModel has been evaluated on the FUNSD dataset, achieving competitive results in extracting meaningful information from challenging, real-world documents.

Evaluation Loss: 1.36752
F1-Score: 0.84126

Installation
To install and use DocModel, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/tobiadefami/docmodel.git
cd docmodel
```

2. Install the package using setup.py:
```bash
python setup.py install
```
3. You can also install the model dependencies using pip:
```bash
pip install -r requirements.txt
```

### Applications
DocModel can be used for a variety of document understanding tasks, including:

1. Form understanding: Extracting key-value pairs from structured forms.
2. Entity extraction: Identifying important information from documents with diverse layouts.
3. Layout-based analysis: Handling complex layouts involving tables, scanned images, and multi-column formats.

### Model Availability

Model Hub: DocModel on [Hugging Face Hub](https://huggingface.co/tobiadefami/docmodel-base)


### License
This project is licensed under the Mozilla Public License 2.0. See the LICENSE file for details.

### Contact
For any questions or inquiries, feel free to reach out!
