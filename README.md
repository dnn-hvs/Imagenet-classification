# Imagenet-classification

Has code to test models against the ImageNet dataset

## Usage

**Note: Run the `prepare_val.py` script inside the validation folder to make sure everything is in order.**

### To run for pre-trained models

```bash
python3 main.py [PATH to ImageNet] --pretrained
```

### To run for all models in a directory

```bash
python3 main.py [PATH to ImageNet] --model_dir [Path to models dir] --save_file test
```
