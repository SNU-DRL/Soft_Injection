# Soft Injection of Task Embeddings Outperforms Prompt-Based In-Context Learning
Official Pytorch Implementation of "Soft Injection of Task Embeddings Outperforms Prompt-Based In-Context Learning" 

### [Paper Link](https://arxiv.org/abs/2507.20906)


## Setup

This code was tested with Python 3.12.2 using A6000 GPU.

To construct the environment, run the following command:
``` python
conda env create -f site_environment.yml
```

## Quickstart
1. Run the following command to activate environment:
   ``` python
   conda activate site
   ```
2. Run your script file in `./eval_scripts` folder.


## Acknowledgments

This project builds on the codes from the following repositories:

- [Function Vector](https://github.com/ericwtodd/function_vectors)
- [Multimodal Task Vector](https://github.com/Brandon3964/MultiModal-Task-Vector)

We thank the creators of these projects for making their codes available.
