# Benchmarking

This page describes the benchmark scripts that can be used to measure the performance impact of different changes to
hivemind.

### Server throughput

You can use [this benchmark](https://github.com/learning-at-home/hivemind/blob/master/tests/benchmark_throughput.py) to
check the performance impact of your changes to hivemind.client and server. The benchmark will start one server without
DHT with several experts, and then spawn trainer processes that load the server with requests. The two main statistics
in this benchmark samples/s and startup time.

`python benchmark_throughput.py --preset default` (aka `ffn_forward_backward`)

<details style="margin-top:-24px; margin-bottom: 16px;">
  <summary>Console outputs</summary>

  ```sh
Benchmark finished, status:Success
Server parameters: num_experts=16, num_handlers=64, max_batch_size=8192, expert_cls=ffn, hid_dim=1024, device=cuda
Client parameters: num_clients=128, num_batches_per_client=16, batch_size=2048, backprop=True
Results: 
	Server startup took 10.965 s. (3.075 s. experts + 7.889 s. networking)
	Processed 4194304 examples in 146.750
	Throughput for forward + backward passes: 28581.213 samples / s.
	Benchmarking took 157.948 s.
Using device: cuda
GeForce GTX 1080 Ti
Memory Usage:
Allocated: 6.0 GB
Cached:    7.7 GB

  ```

</details>

`python benchmark_throughput.py --preset ffn_forward`

<details style="margin-top:-24px; margin-bottom: 16px;">
  <summary>Console outputs</summary>

  ```sh
Benchmark finished, status:Success
Server parameters: num_experts=16, num_handlers=64, max_batch_size=8192, expert_cls=ffn, hid_dim=1024, device=cuda
Client parameters: num_clients=128, num_batches_per_client=16, batch_size=2048, backprop=False
Results: 
	Server startup took 19.941 s. (3.065 s. experts + 16.877 s. networking)
	Processed 4194304 examples in 42.973
	Throughput for forward passes: 97604.282 samples / s.
	Benchmarking took 63.167 s.
Using device: cuda
GeForce GTX 1080 Ti
Memory Usage:
Allocated: 1.5 GB
Cached:    3.2 GB
```

</details>

### DHT performance

In turn, [this benchmark](https://github.com/learning-at-home/hivemind/blob/master/tests/benchmark_dht.py) can be used
to measure performance impact of changes to hivemind.dht. It spawns a DHT with `num_peers` participants, then chooses
one peer that will declare `num_experts` total experts in batches of `expert_batch_size`. Then, another peer will
consecutively get all peers and check if they are there.

Here's a run with 1024 participants on the same machine that was used for benchmark_throughput:

`python benchmark_dht.py --num_peers 1024 --num_experts 16384 --expert_batch_size 64 --expiration 99999 --increase_file_limit`
<details style="margin-top:-24px; margin-bottom: 16px;">
  <summary>Console outputs</summary>

  ```sh
Increasing file limit - soft 1024=>32768, hard 1048576=>32768
Creating peers...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [01:45<00:00,  9.74it/s]
Sampled 16384 unique ids (after deduplication)
Storing peers to dht in batches of 64...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 256/256 [12:07<00:00,  2.84s/it]
Store success rate: 100.0% (48920 / 48920)
Mean store time: 0.01487, Total: 727.46
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 256/256 [01:48<00:00,  2.35it/s]
Get success rate: 100.0 (16384 / 16384)
Mean get time: 0.00664, Total: 108.73952
Node survival rate: 100.000%
  ```

</details>

The three main statistics in this benchmark are total store time, total get time and get success rate. Please also note
that this benchmark does not emulate node failure, latency and does not benefit from caching. If one wants to account
for these factors, one must introduce them manually by changing the code.