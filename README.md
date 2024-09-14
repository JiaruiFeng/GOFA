# Generative One-For-All (GOFA)
 The source code for paper [GOFA: A  generative one-for-all model for joint graph language modeling](https://arxiv.org/abs/2407.09709). The code is still under clean. Feel free to open an issue in GitHub if you encounter any problem. 

## Installation Guide.
First, clone the code repository and move to the code file. Then, create the python environment. We provide environment configuration:
```
conda env create -f environment.yml
```

Next, please download the pretrained checkpoint of ICAE from [here](https://huggingface.co/sggetao/icae/tree/main). Specifically, download and put
the following files `llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt` and `mistral_7b_ft_icae.safetensors` into directory `./cache_data/model/`

Finally, clone the code of datasets we used from [TAGLAS](https://github.com/JiaruiFeng/TAGLAS) by running:
```
git clone https://github.com/JiaruiFeng/TAGLAS.git
```


## Pre-training
Pre-training require large computation resource and time. If you want to explore GOFA, we recommend you to download our pre-trained checkpoints and directly run downstream fine-tuning. You can download checkpoints from [here](https://huggingface.co/WFRaain/GOFA/tree/main).
We provide checkpoints for both Llama2 (`qamag03_best_ckpt.pth`) and Mistral (`mistral_qamag03_best_ckpt.pth`). 

To run the pretraining by yourself, please first generate pretraining data using the following script. 

```
python pretrain_data_generation.py
```
The above code will generate three pretrain data subset. Note that the generation process require huge memory and will last for long time. Please allocate enough resource for generation.

After data generation, run the following line to start the pretraining:
```
python run_gofa.py --override ./configs/pretrain_config.yaml
```
This code will run pretraining of the GOFA llama2 version on the first pretrain data subset. if you want to train the mistral version, run:
```
python run_gofa.py --override ./configs/pretrain_config.yaml base_llm mistral7b
```
To continue the training on next subset, check the `last_epochs` in the `./configs/default_config.yaml` to the next batch and `ckpt_path` to the saved checkpoint on the last pretraining.

## Instruction fine-tuning for zero-shot experiment.
To repeat the experiments of GOFA on zero-shot learning with arxiv instruction tuning, run:
```
python run_gofa.py --override ./configs/instruct_arxiv_config.yaml load_dir llama_pretrained_model_pth base_llm llama7b
python run_gofa.py --override ./configs/instruct_arxiv_config.yaml load_dir mistral_pretrained_model_pth base_llm mistral7b
```
Please change the `load_dir` to either the corresponding downloaded checkpoints or your own pretrained checkpoints.

Similarly, to repeat the experiments of GOFA on zero-shot learning with pubmed link instruction tuning, run:
```
python run_gofa.py --override ./configs/instruct_pubmed_config.yaml load_dir mistral_pretrained_model_pth base_llm mistral7b
```

## Supervised fine-tuning.
To repeat the experiments of GOFA on supervised learning, run:
```
python run_gofa.py --override ./configs/supervised_config.yaml load_dir pretrained_model_pth base_llm model_type
```
Similar as the above, modify the `load_dir` and `base_llm` for validating corresponding model.

## Evaluation and inference
To explore the generation result of GOFA, you can also directly run the inference mode with: 
```
python run_gofa.py --override ./configs/inference_config.yaml load_dir finetuned_model_pth base_llm llama7b
```
Please modify the config file for selecting corresponding dataset. Note that for both zero-shot and supervised experiment, the
trained model should be evaluated under inference model to obtain the correct evaluation result. 

We also provide checkpoint of mistral version of GOFA on arxiv instruction-tuning in [here](https://huggingface.co/WFRaain/GOFA/tree/main) with file name `nb_instruct.pth`. 
You can use this checkpoint to directly reproduce the results in the paper. 

## Citation
```
@article{kong2024gofa,
  title={GOFA: A Generative One-For-All Model for Joint Graph Language Modeling},
  author={Kong, Lecheng and Feng, Jiarui and Liu, Hao and Huang, Chengsong and Huang, Jiaxin and Chen, Yixin and Zhang, Muhan},
  journal={arXiv preprint arXiv:2407.09709},
  year={2024}
}
```

