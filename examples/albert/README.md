# Training ALBERT

This tutorial will walk you through setting up training ALBERT-large-v2 with WikiText103 dataset. It will walk you through the necessary steps to setup your own collaborative training for similar models and tasks.

### Preparation
* Install hivemind: `pip install git+https://github.com/learning-at-home/hivemind.git`
* Dependencies: `pip install "transformers>=4.5.1" "datasets>=1.5.0" "torch_optimizer>=0.1.0" sentencepiece wandb whatsmyip`
* Preprocess data: `python ./tokenize_wikitext103.py`
* Upload an archive preprocessed data to somewhere volunteers can reach, example: `https://hivemind-data.s3.us-east-2.amazonaws.com/wikitext103_preprocessed.tar`


## Start training
- Run the first DHT peer (aka "coordinator") on a node that is accessible to all trainers: `python run_first_peer.py --listen_on [::]:1337` (see details below)
- For all GPU trainers, run

```python run_trainer.py \
  --output_dir ./outputs --overwrite_output_dir \
  --logging_dir ./logs --logging_first_step --logging_steps 100 \
  --initial_peers COORDINATOR_IP:COORDINATOR_PORT --seed 0
```


The coordinator node exists solely to welcome other peers onto the DHT. It requires neither GPU nor high bandwidth, the only prerequisite is that coordinator should have high uptime. If no high uptime server is available, one can also run multiple coordinators on different servers and list all of them as --initial_peers. The system will work as long as at least one coordinator is available.

The trainer node can be launched on any computer with a GPU, such as AWS VM or vast.ai instance. Trainer nodes can be added to the system at any time.

`-----------------`
To be added:
- data preprocessing
- how to run (first peer, trainers)
- tips on how to host a collaboration (colab/kaggle runners, client_mode)
- explain collaboration metrics
- how to read/extend the code
- data uploading best practices (s3 vs local server vs dropbox vs preprocess independently)
