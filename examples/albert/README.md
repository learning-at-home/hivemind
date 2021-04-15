# Training ALBERT

This tutorial will walk you through setting up training ALBERT-large-v2 with WikiText103 dataset

### Preparation
* Install hivemind: `pip install git+https://github.com/learning-at-home/hivemind.git`
* Dependencies: `pip install "transformers>=4.5.1" "datasets>=1.5.0" "torch_optimizer>=0.1.0" sentencepiece wandb whatsmyip`
* Preprocess data: `python ./tokenize_wikitext103.py`
* Upload an archive preprocessed data to somewhere volunteers can reach, example: `https://hivemind-data.s3.us-east-2.amazonaws.com/wikitext103_preprocessed.tar`


## Start training


To be added:
- data preprocessing
- how to run (first peer, trainers)
- tips on how to host a collaboration (colab/kaggle runners, client_mode)
- explain collaboration metrics
- how to read/extend the code
- data uploading best practices (s3 vs local server vs dropbox vs preprocess independently)
