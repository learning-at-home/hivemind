## Contributing

#### Collaborating best practices:
Hivemind is still in the early stage of development, we expect only a handful of collaborators with individual roles.

1. Before you write any code, please contact us to avoid duplicate work:
   * Report bugs and propose new features via issues. We don't have strict templates at this point;
   * If you decide to implement a feature or fix a bug, first leave a comment in the appropriate issue or create a
    new one;
   * Please follow [Contributor Convent v2.0](https://www.contributor-covenant.org/version/2/0/code_of_conduct/).
2. When you code, follow the best practices:
   * The code must follow [PEP8](https://www.python.org/dev/peps/pep-0008/) unless absolutely necessary.
     We recommend pycharm IDE;
   * All user-facing interfaces must be documented with docstrings and/or sphinx;
   * We highly encourage the use of [typing](https://docs.python.org/3/library/typing.html), where applicable;
3. After you write the code, make sure others can use it:
   * Any function exposed to a user must have a docstring compatible with [readthedocs](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html);
   * For new features, please write test(s) to make sure your functionality won't be broken by subsequent changes;
   * If you face any challenges or want feedback, please submit a [draft](https://github.blog/2019-02-14-introducing-draft-pull-requests/) pull request.


#### Contributor's manual

First, install hivemind in the development mode, preferably with python 3.8 on linux/mac_OS.
```
git clone https://github.com/learning-at-home/hivemind
cd hivemind
python setup.py develop
``` 

To run tests, you will also need to `pip install pytest codecov tqdm scikit-learn`.
You can run all tests with `pytest ./tests` or choose a specific set, e.g. `pytest ./tests/test_dht.py`.

To build docs locally,
1. `pip install sphinx sphinx_rtd_theme recommonmark`
2. make sure you ran setup.py (see above)
3. `cd ./docs && make html`

The documentation root will be available in `./docs/_build/html/index.html`


#### Benchmark throughput
You can use [this benchmark](https://github.com/learning-at-home/hivemind/blob/master/tests/benchmark_throughput.py) to check the performance impact of your changes to hivemind.client and server.
The benchmark will start one server without dht with several experts, and then spawn trainer processes that bombard the server with requests.
The two main statistics in this benchmark samples/s and startup time. 

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

All tests were performed on a single machine with ubuntu server 18.04 x64, msi 1080ti turbo, xeon gold 6149, 
 384Gb LRDIMM (6x64G), python3.8, torch1.6.0 (pip-installed), grpcio 1.31.0 , 
 the results have around +-5% fluctuation between consecutive runs. 

#### Benchmark DHT
In turn, [this benchmark](https://github.com/learning-at-home/hivemind/blob/master/tests/benchmark_dht.py) can be used
to measure performance impact of changes to hivemind.dht. It spawns a DHT with `num_peers` participants, 
then chooses one peer that will declare `num_experts` total experts in batches of `expert_batch_size`.
Then, another peer will consecutively get all peers and check if they are there.

Here's a run with 1024 participants on the same machine that was used benchmark_throughput:

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

The three main statistics in this benchmark are total store time, total get time and get success rate.
Please also note that this benchmark does not emulate node failure, latency and does not benefit from caching.
If one wants to account for these factors, one must introduce them manually by changing the code.
  

#### Tips & tricks
* You can find a wealth of pytorch debugging tricks at [their contributing page](https://tinyurl.com/pytorch-contributing).
* Hivemind is optimized for development in pycharm CE 2019.3 or newer.
  * When working on tests, please mark "tests" as sources root.
