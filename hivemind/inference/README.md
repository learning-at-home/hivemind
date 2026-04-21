# Decentralized AI Inference Module

## Overview

This module enables running large language models (like Kimi K2.6) across distributed GPU/CPU resources contributed by participants worldwide. It implements three core innovations:

1. **Shared Compute Pool**: Multiple users contribute their consumer GPUs/CPUs to collectively run massive models
2. **Token-Based Rewards**: Contributors earn tokens proportional to their compute contribution  
3. **User-Friendly Interface**: Simple CLI commands for non-technical users

## Key Features

### 1. Pipeline Parallelism
Split large models into chunks that run across different nodes globally. A model with 100 layers can be distributed so that:
- User A runs layers 0-10 on their RTX 4090
- User B runs layers 11-25 on their A100
- User C runs layers 26-40 on their CPU
- And so on...

### 2. Contribution Tracking
Automatically tracks compute time, tokens processed, and device type to calculate fair rewards distribution.

### 3. Smart Resource Discovery
DHT-based protocol discovers available compute resources and assembles complete inference pipelines.

## Installation

```bash
pip install hivemind
```

## Quick Start

### Contribute Your GPU

Contribute layers 0-4 of Kimi K2.6 model:

```bash
hivemind-inference contribute --model kimi-k2.6 --layers 0-4 --track-rewards
```

Or specify a different layer range:

```bash
hivemind-inference contribute --model kimi-k2.6 --layers 20-30 --device cuda
```

### Run a Large Model

Run inference using the distributed compute pool:

```bash
hivemind-inference run --model kimi-k2.6 --prompt "What is quantum computing?" --max-tokens 100
```

With verbose output:

```bash
hivemind-inference run --model kimi-k2.6 --prompt "Explain machine learning" --verbose
```

### Check Status

View your contributions:

```bash
hivemind-inference status
```

### Discover Resources

Check what models are available:

```bash
hivemind-inference discover --model kimi-k2.6 --coverage
```

## Python API

### Contribute Compute Resources

```python
from hivemind import DHT, ModelChunkProvider, ModelChunkConfig
from hivemind.inference.contribution import ContributionTracker

# Initialize DHT
dht = DHT(initial_peers=[], start=True)

# Configure your contribution
config = ModelChunkConfig(
    model_name="kimi-k2.6",
    layer_start=0,
    layer_end=4,
    device="cuda",  # or "cpu"
    hidden_dim=4096
)

# Create provider
provider = ModelChunkProvider(dht, config)
provider.advertise()

# Track contributions for rewards
tracker = ContributionTracker(dht)
tracker.run_in_background()

# Keep running
try:
    while True:
        time.sleep(10)
        stats = provider.get_stats()
        print(f"Compute time: {stats['total_compute_time']:.2f}s")
except KeyboardInterrupt:
    dht.shutdown()
```

### Run Distributed Inference

```python
import asyncio
from hivemind import DHT, PipelineParallelRunner

async def main():
    dht = DHT(start=True)
    runner = PipelineParallelRunner(dht, "kimi-k2.6")
    
    # Discover available compute
    topology = await runner.discover_topology()
    print(f"Found {len(topology)} model chunks")
    
    # Run inference
    result = await runner.generate("What is AI?", max_tokens=50)
    print(result)
    
    dht.shutdown()

asyncio.run(main())
```

### Track Rewards

```python
from hivemind import DHT, ContributionTracker, TokenRewardCalculator

dht = DHT(start=True)
tracker = ContributionTracker(dht)

# Start tracking a session
session = tracker.start_session(
    session_id="session_123",
    model_name="kimi-k2.6",
    total_expected_tokens=1000
)

# Record contributions
tracker.record_contribution(
    session_id="session_123",
    expert_uid="kimi-k2.6.layer.0.4",
    layer_range=(0, 4),
    compute_time=2.5,
    tokens_processed=100,
    device_type="gpu"
)

# Finalize and get rewards
rewards = tracker.finalize_session("session_123", actual_tokens=950)
print(f"Rewards distribution: {rewards}")

# Calculate monetary rewards
calculator = TokenRewardCalculator(tracker)
actual_rewards = calculator.distribute_rewards("session_123", total_token_value=100.0)
```

## Architecture

### Components

1. **ModelChunkProvider**: Hosts a chunk of a large model
2. **PipelineParallelRunner**: Assembles and runs distributed inference
3. **ContributionTracker**: Records and tracks compute contributions
4. **TokenRewardCalculator**: Calculates fair reward distribution
5. **LayerDiscoveryProtocol**: Discovers available compute resources
6. **ResourceRegistry**: Maintains registry of available resources

### How It Works

1. **Contribution Phase**:
   - Users run `contribute` command specifying which model layers to host
   - Their compute resources are advertised to the DHT network
   - Contribution tracker monitors compute time and tokens processed

2. **Discovery Phase**:
   - Runner queries DHT for available model chunks
   - Assembles complete pipeline from available resources
   - Prefers GPU resources, fills gaps with CPU

3. **Inference Phase**:
   - Input passes through each layer chunk sequentially
   - Each contributor's compute time is recorded
   - Results are streamed back to the user

4. **Reward Phase**:
   - Session is finalized with actual token count
   - Rewards calculated based on contribution percentage
   - Contributors receive tokens proportional to their compute

## Reward System

Contributions are weighted by:
- **Compute Time**: Longer computation = more rewards
- **Device Type**: GPU compute weighted 2x vs CPU
- **Tokens Processed**: More tokens = more value contributed

Formula:
```
compute_units = compute_time × device_weight
reward_percentage = your_compute_units / total_compute_units
your_tokens = total_tokens × reward_percentage
```

## Advanced Usage

### Custom Models

Add support for custom models by registering expert classes:

```python
from hivemind import register_expert_class
import torch.nn as nn

class CustomTransformer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Your custom architecture
    
    def forward(self, x):
        return x

register_expert_class("custom_transformer", CustomTransformer)
```

### Multiple Layer Ranges

Run multiple providers on the same machine:

```bash
# Terminal 1
hivemind-inference contribute --layers 0-9

# Terminal 2
hivemind-inference contribute --layers 10-19

# Terminal 3
hivemind-inference contribute --layers 20-29
```

### Production Deployment

For production use with specific peers:

```bash
hivemind-inference contribute \
  --initial-peers /ip4/1.2.3.4/tcp/8000/p2p/QmPeer1 \
  --track-rewards
```

## Troubleshooting

### No Resources Found

If `run` command finds no resources:
1. Ensure at least one node is contributing layers
2. Check DHT connectivity with `discover --coverage`
3. Verify initial peers are correct

### Slow Inference

Performance depends on:
- Network latency between nodes
- Number of gaps in pipeline
- Device types (GPU vs CPU)

Optimize by:
- Contributing consecutive layer ranges
- Using GPU when possible
- Reducing number of unique peers in pipeline

### Contribution Not Tracked

Ensure `--track-rewards` flag is set when contributing. Check logs for:
```
📊 Enabling contribution tracking...
Recorded contribution: X.XXXs on gpu for layers (X, Y), Z tokens
```

## Future Enhancements

Planned improvements:
- [ ] Blockchain integration for token distribution
- [ ] Automatic load balancing
- [ ] Fault tolerance with checkpointing
- [ ] Dynamic layer reassignment
- [ ] Performance benchmarking dashboard
- [ ] Web UI for monitoring
- [ ] Mobile app for contribution tracking

## Security Considerations

- All tensor data is encrypted during transmission
- Contribution records are publicly verifiable via DHT
- No model weights are exposed to intermediate nodes
- Peer reputation system (planned) to prevent abuse

## Citation

If you use this in your research:

```bibtex
@software{hivemind_inference,
  title = {Hivemind Decentralized Inference},
  author = {Hivemind Team},
  year = {2024}
}
```

## License

Same license as Hivemind project.
