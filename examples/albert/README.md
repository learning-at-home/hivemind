# Training ALBERT with decentralized averaging

This tutorial will walk you through the steps to set up collaborative training with the ALBERT-large-v2 model and the WikiText103 dataset. It uses huggingface [datasets](https://github.com/huggingface/datasets) and [transformers](https://github.com/huggingface/transformers/) libraries to compute local updates, using `hivemind.CollaborativeOptimizer` to exchange information between peers.

### Preparation
* Install hivemind: `pip install git+https://github.com/learning-at-home/hivemind.git`
* Dependencies: `pip install -r requirements.txt`
* Preprocess data: `python tokenize_wikitext103.py`
* Upload an archive preprocessed data to somewhere volunteers can reach, example: `https://hivemind-data.s3.us-east-2.amazonaws.com/wikitext103_preprocessed.tar`


## Running an experiment
- Run the first DHT peer to welcome trainers and record training statistics (e.g. loss, performance):
   - In this example, we use [wandb.ai](https://wandb.ai/site) to plot training metrics; If you're unfamiliar with Weights & Biases, here's a [quickstart tutorial](https://docs.wandb.ai/quickstart).
   - Run `python run_training_monitor.py --dht_listen_on '[::]:*' --experiment_prefix NAME_YOUR_EXPERIMENT --wandb_project WANDB_PROJECT_HERE`
   - `NAME_YOUR_EXPERIMENT` must be a unique name of this training run, e.g. `my-first-albert`. It cannot contain `.` due to naming conventions.
   - `WANDB_PROJECT_HERE` is a name of wandb project used to track training metrics. Multiple experiments can have the same project name.
   - This peer will run a DHT node on a certain IP/port (`Running DHT root at ...`). You will need this address for next steps
```
+ python run_training_monitor.py --dht_listen_on '[::]:*' --experiment_prefix my-albert-v1 --wandb_project Demo-run
[2021/06/17 16:26:35.931][WARN][root.<module>:140] No address specified. Attempting to infer address from DNS.
[2021/06/17 16:26:36.083][INFO][root.<module>:149] Running DHT root at 193.106.95.184:38319
wandb: Currently logged in as: XXX (use `wandb login --relogin` to force relogin)
wandb: Tracking run with wandb version 0.10.32
wandb: Syncing run dry-mountain-2
wandb:  View project at https://wandb.ai/XXX/Demo-run
wandb:  View run at https://wandb.ai/XXX/Demo-run/runs/YYY
wandb: Run data is saved locally in /path/to/run/data
wandb: Run `wandb offline` to turn off syncing.
[2021/04/19 02:26:41.064][INFO][optim.collaborative.fetch_collaboration_state:323] Found no active peers: None
[2021/04/19 02:26:44.068][INFO][optim.collaborative.fetch_collaboration_state:323] Found no active peers: None
...
[2021/04/19 02:37:37.246][INFO][root.<module>:74] 11.05164
[2021/04/19 02:39:37.441][INFO][root.<module>:74] 11.03771
[2021/04/19 02:40:37.541][INFO][root.<module>:74] 11.02886
```

- To join a collaboration with a GPU trainer, 
  - install the same dependencies (minus the `wandb` and `whatsmyip`), download the data and unpack it to the experiment folder,
  - if necessary, specify paths: `--dataset_path ./path/to/unpacked/data --tokenizer ./path/to/tokenizer/config` (see [default paths](https://github.com/learning-at-home/hivemind/blob/collaborative_albert_example/examples/albert/run_trainer.py#L63-L69) for reference)
  - run:
```shell
python run_trainer.py \
 --experiment_prefix SAME_AS_IN_RUN_TRAINING_MONITOR --initial_peers ONE_OR_MORE_PEERS --seed 42 \
 --logging_first_step --logging_steps 100  --output_dir ./outputs --overwrite_output_dir --logging_dir ./logs
```
Here, `ONE_OR_MORE_PEERS` stands for either your coordinator endpoint (e.g. `123.123.123.123:1337`), an endpoint of any pre-existing trainer or multiple endpoints for stability. See tips & tricks section below for more information on setting up collaborative training.

As the peer begins training, it will periodically report training logs in the following form:
```
[...][INFO][...] Collaboration accumulated 448 samples from 17 peers; ETA 18.88 seconds (refresh in 15.73s.)
[...][INFO][...] Collaboration accumulated 4096 samples from 16 peers; ETA 0.00 seconds (refresh in 0.50s.)
[...][INFO][optim.collaborative.step:195] Averaged tensors successfully with 17 peers
[...][INFO][optim.collaborative.step:211] Optimizer step: done!
06/17/2021 18:58:23 - INFO - __main__ -   Step 0
06/17/2021 18:58:23 - INFO - __main__ -   Your current contribution: 892 samples
06/17/2021 18:58:23 - INFO - __main__ -   Local loss: 11.023

```

__Sanity check:__ a healthy peer will periodically report `Averaged tensors successfully with [N > 1]` peers.

For convenience, you can view (and share!) the learning curves of your collaborative experiments in wandb:
![image](https://user-images.githubusercontent.com/3491902/115177859-bed5e100-a0d8-11eb-82bc-55d1b12d335d.png)


## Tips and tricks

Finally, we provide best practices for running collaborative experiments of different sizes.

### Hosting the data
For small experiments (3-16 peers, <1GB data), you can use a free-tier file hosting that has a convenient way to [download with curl/wget](https://superuser.com/questions/470664/how-to-download-dropbox-files-using-wget-command). However, these services are not meant for high load and could ban you for generating too much traffic. If you want to scale up, you could either use an S3-like storage from [any](https://aws.amazon.com/s3/) [cloud](https://cloud.google.com/storage) [provider](https://cloud.google.com/storage) or host the data [yourself]((https://gist.github.com/willurd/5720255)). Large data files (>5GB) will take long to download; we recommend splitting them into chunks and implementing a custom dataloader that can load chunks on the fly. Finally, the most _comme il faut_ solution to sharing large datasets is to use [academic torrents](https://academictorrents.com/).
 
### run_training_monitor.py
This peer exists solely to welcome other peers onto the DHT and track learning progress. It requires neither GPU nor high bandwidth, the only prerequisite is that coordinator should have high uptime. If no high uptime server is available, one can also run multiple coordinators on different servers and list all of them as `--initial_peers`. The system will stay up as long as at least one coordinator is available. For short- to mid-term experiments you can host coordinator on a [free-tier VM](https://www.quora.com/Are-there-any-free-online-virtual-machines).

### Tuning for hardware/network
The optimal training parameters for each peer depend on its GPU and internet connection. If a peer cannot accept incoming connections (e.g. when in colab or behind a firewall), add `--client_mode` to the training script (see example below). In case of high network latency, you may want to increase `--averaging_expiration` by a few seconds or set `--batch_size_lead` to start averaging a bit earlier than the rest of the collaboration. GPU-wise, each peer should be able to process one local microbatch each `0.5~1` seconds (see trainer's progress bar). To achieve that, we recommend tuning `--per_device_train_batch_size` and `--gradient_accumulation_steps`. The example trainer supports multiple GPUs via DataParallel. However, using advanced distributed training strategies (e.g. [ZeRO-3](https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html)) will require changes in `run_trainer.py`.

### Using public GPU providers
There are awesome services like [Google Colab](https://colab.research.google.com/), [Kaggle kernels](https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu) or[Paperspace](https://gradient.paperspace.com/free-gpu) that provide free GPUs. These services usually come with significant limitations (e.g. last gen GPUs, reset every few hours), but they allow just about anyone to join your collaborative experiment. Here's how to best use them.
  - before you begin, __read the rules carefully__. Most free-tier GPU services allow only one GPU per user and using more than one account will get you banned. It is **your** duty to make sure that collaborators won't get in trouble for helping you.
  - most free GPUs are running behind a firewall, which requires you to run trainer with `--client_mode` (see example below). Such peers can only exchange gradients if there is at least one non-client-mode peer (GPU server or desktop with public IP). We recommend using a few preemptible instances with the cheapest GPU you can find. For example, we tested this code on preemptible [`g4dn.xlarge`](https://aws.amazon.com/blogs/aws/now-available-ec2-instances-g4-with-nvidia-t4-tensor-core-gpus/) nodes for around $0.15/h apiece with 8 AWS nodes and up to 61 Colab/Kaggle participants.
  - you can create starter notebooks to make it more convenient for collaborators to join your training run ([example](https://colab.research.google.com/gist/yhn112/e858cb841c73879d8ef98a84e03b43e7/collaborative-training-v0-10.ipynb)). Ideally, joining collaboration should take at most a couple of clicks.

Here's an example of a full trainer script for Google Colab:
```
!pip install transformers datasets sentencepiece torch_optimizer==0.1.0
!git clone https://github.com/learning-at-home/hivemind && cd hivemind && pip install -e .
!curl -L YOUR_HOSTED_DATA | tar xzf -     # example: https://hivemind-data.s3.us-east-2.amazonaws.com/wikitext103.tar.gz
!ulimit -n 4096 && python ./hivemind/examples/albert/run_trainer.py \
 --client_mode --initial_peers ONE_OR_MORE_PEERS  --averaging_expiration 10 \
 --batch_size_lead 300 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
 --logging_first_step --logging_steps 100  --output_dir ./outputs --overwrite_output_dir --logging_dir ./logs \
 --experiment_prefix EXPERIMENT_NAME_HERE --seed 42
```
