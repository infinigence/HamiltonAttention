# Test Script Usage

- [Test Script Usage](#test-script-usage)
  - [Hamilton Attention Test Script Usage](#hamilton-attention-test-script-usage)
    - [Basic Testing](#basic-testing)
    - [Example Test Configuration](#example-test-configuration)
  - [Test Outputs](#test-outputs)
    - [Performance Metrics](#performance-metrics)
    - [Accuracy Validation](#accuracy-validation)
  - [Advanced Testing Features](#advanced-testing-features)
    - [1. Profiling Modes](#1-profiling-modes)
      - [Performance Profiling](#performance-profiling)
      - [PyTorch Profiler Integration](#pytorch-profiler-integration)
    - [2. Trace Export](#2-trace-export)
  - [Running Tests](#running-tests)
    - [Single-node Test](#single-node-test)
    - [Multi-node Test](#multi-node-test)
  - [Configuration Parameters](#configuration-parameters)
    - [Model Dimensions](#model-dimensions)
    - [Test Configuration](#test-configuration)
    - [Advanced Options](#advanced-options)


## Hamilton Attention Test Script Usage

### Basic Testing

The main testing function `prof_perform()` provides a complete benchmarking framework:

```python
def prof_perform(
    bs,                    # Batch size
    seqlen,               # Sequence length  
    nheads,               # Number of attention heads
    d,                    # Head dimension (typically 64)
    causal,               # Whether to use causal attention
    two_nodes,            # Whether to use (m-K_m-m)^u-decomposition scheme
    test_nums,            # Number of test iterations
    warm_ups,             # Number of warm-up iterations
    local_rank,           # Local GPU rank
    log_all=False         # Whether to log detailed outputs
)
```

### Example Test Configuration

```python
# Single node, non-causal attention test
results = prof_perform(
    bs=2,                 # Batch size 2
    seqlen=4096,          # Sequence length 4096
    nheads=16,            # 16 attention heads
    d=64,                 # Head dimension 64
    causal=False,         # Non-causal attention
    two_nodes=False,      # Use K_{m\times u}-decomposition scheme
    test_nums=10,         # 10 test iterations
    warm_ups=3,           # 3 warm-up iterations
    local_rank=0,         # GPU 0
    log_all=True          # Enable detailed logging
)
```


## Test Outputs

### Performance Metrics

The test function returns detailed performance data including:

```python
{
    "hamilton": {
        "name": "K8-hamilton",           # Test configuration name
        "batch_size": 2,                 # Batch size
        "seqlen": 4096,                  # Sequence length
        "nheads": 16,                    # Number of heads
        "d": 64,                         # Head dimension
        "causal": False,                 # Attention type
        "two_nodes": False,              # Decomposition scheme flag
        "test_nums": 10,                 # Test iterations
        "warm_ups": 3,                   # Warm-up iterations
        "data": [                        # Performance data for each iteration
            {
                'total_time': 45.2,      # Total execution time (ms)
                'comm_total_time': 12.1, # Total communication time
                'comp_total_time': 33.1  # Total computation time
            },
            # ... more iterations
        ]
    },
    "ring": {
        # Similar structure for ring/zig-zag ring attention results
    }
}
```

### Accuracy Validation

When `accuracy_test=True`, the framework validates numerical correctness:

```python
# Cosine similarity between distributed and reference outputs
cosine_sim_out = F.cosine_similarity(
    torch.flatten(local_out), 
    torch.flatten(dist_out[iter0]), 
    dim=0
)
```

## Advanced Testing Features
### 1. Profiling Modes

#### Performance Profiling
```python
# Enable detailed timing breakdown
profile_version = True
```

#### PyTorch Profiler Integration
```python
# Enable PyTorch profiler for detailed operator analysis
torch_profile_on = True
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
prof = torch.profiler.profile(activities=activities, schedule=schedule)
```

### 2. Trace Export

Traces and prof_results are exported for every rank. The default root dir of trace and prof_results output are the dir of this test script itself.
```python
# Export Chrome trace for visualization in TensorBoard
prof.export_chrome_trace(f"{path_prefix}/trace_{config}_{local_rank}.json")
```

## Running Tests

### Single-node Test

```bash
# Simple Node test
torchrun --nproc_per_node=8 test_hamilton_attention.py \
```

### Multi-node Test

```bash
# 16-GPU two-node test  
torchrun --nnodes 2 --nproc_per_node=8 test_hamilton_attention.py \
```

## Configuration Parameters

### Model Dimensions
| Parameter | Typical Values | Description |
|-----------|----------------|-------------|
| `bs` | 1-128 | Batch size |
| `seqlen` | 1K-1M | Sequence length (must be divisible by world_size) |
| `nheads` | 8-128 | Number of attention heads |
| `d` | 64, 128 | Head dimension |

**⚠️ NOTICE**:
- **For $K_{m\times u}$-decomposition, seqlen must be divisible by world_size\*(world_size-1)**
- **For $(m-K_m-m)^u$-decomposition, seqlen must be divisible by world_size\*($m$)**
- These restrictions exist due to Hamilton Attention need to split the sequence into more chunks (compared to Ring Attention) to fully utilize all the available communication links (see [paper](https://arxiv.org/abs/2509.26541)).  

### Test Configuration
| Parameter | Options | Description |
|-----------|---------|-------------|
| `causal` | True/False | Causal or non-causal attention |
| `two_nodes` | True/False | Use $(m-K_m-m)^u$-decomposition scheme |
| `test_nums` | 4-50 | Number of test iterations |
| `warm_ups` | 3-10 | Warm-up iterations before timing |

### Advanced Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `accuracy_test` | False | Enable numerical accuracy validation |
| `profile_version` | True | Enable performance profiling |
| `torch_profile_on` | False | Enable PyTorch profiler |
| `log_all` | False | Enable detailed logging |