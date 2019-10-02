# Automatic icon alt-text

### Summary
Automatically providing alt text to graphical symbols by doing nearest neighbor search on simple shape image descriptors. Every nearest neighbor candidate has an alt text associated with it. The dataset is automatically created by using the method I implemented here: https://github.com/OliverEdholm?tab=repositories

### Running
#### Building model
`python3 -m bin.build_auto_alt_text_model WEB_ICON_DATASET_PATH MODEL_SAVE_PATH`
#### Running model
`python3 -m bin.run_auto_alt_text_model INFERENCE_IMAGE_PATH MODEL_PATH`

### Relevant papers
* Conceptual Captions: A New Dataset and Challenge for Image Captioning
* Learning from Simulated and Unsupervised Images through Adversarial Training

### Author
Oliver Edholm, 17 years old
