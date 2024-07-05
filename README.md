# Generative One-For-All (GOFA)
 The source code for paper GOFA: A  generative one-for-all model for joint graph language modeling. The code is still under clean and will come out soon.

## Installation Guide.
First, clone the code repository and move to the code file. Then, create corresponding environment. We provide environment configuration:
```
conda env create -f environment.yml
```

Next, please download the checkpoint of ICAE from [here](https://huggingface.co/sggetao/icae/tree/main). Specifically, download and put
the following files `llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt` and `mistral_7b_pretrained_icae.safetensors` into directory `./cache_data/model/`

Finally, clone the dataset code from [TAGLAS](https://github.com/JiaruiFeng/TAGLAS) by running:
```
git clone https://github.com/JiaruiFeng/TAGLAS.git
```
If you want to reproduce the results of GOFA on fine-tuning tasks, you can download the pretrained checkpoint from [here](https://huggingface.co/WFRaain/GOFA/tree/main)
We provide checkpoint for both Llama2 (`qamag03_best_ckpt.pth`) and Mistral (`mistral_qamag03_best_ckpt.pth`). 

## Pre-training
To reproduce pretraining result, please generate pretraining data using the following script. 

```
python pretrain_data_generation.py
```
The above code will generate three pretrain data subset. Note that the generation process require huge memory and will last for long time. Please allocate enough resource for generation.

After data generation, run the following line to start pretraining:
```
python run_gofa.py
```
This code will run pretraining of the GOFA llama2 version on the first pretrain data subset. if you want to train the mistral version, run:
```
python run_gofa.py base_llm mistral7b
```
To continue the training on next subset, check the `last_epochs` in the `./configs/default_config.yaml` to 1/2 and `ckpt_path` to the saved checkpoint on the last pretraining.

## Instruction fine-tuning for zero-shot experiment.
To reproduce results of GOFA on zero-shot learning with arxiv instruction tuning, run:
```
python run_gofa.py --override ./configs/zs_arxiv_config.yaml load_dir llama_pretrained_model_pth base_llm llama7b
python run_gofa.py --override ./configs/zs_arxiv_config.yaml load_dir mistral_pretrained_model_pth base_llm mistral7b
```
Please change the load_dir to the corresponding downloaded pretrain checkpoints.

To reproduce results of GOFA on zero-shot learning with pubmed link instruction tuning, run:
```
python run_gofa.py --override ./configs/zs_pubmed_config.yaml load_dir mistral_pretrained_model_pth base_llm mistral7b
```

## Supervised fine-tuning.
To reproduce results of GOFA on supervised learning, run:
```
python run_gofa.py --override ./configs/supervised_config.yaml load_dir pretrained_model_pth base_llm model_type
```
Similar as the above, modify the `load_dir` and `base_llm` for validating corresponding model.

## Evaluation and inference
To explore the generation result of GOFA, you can also directly run the inference mode with: 
```
python run_gofa.py --override ./configs/inference_config.yaml load_dir finetuned_model_pth base_llm llama7b
```
Please modify the config file for selecting corresponding dataset.



