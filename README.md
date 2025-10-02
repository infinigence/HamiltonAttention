# Hamilton Attention 

This is the official github repository of Hamilton Attention, also denoted as TASP (Topology-aware Sequence Parallelism).
Our implementation is based on pytorch distributed APIs and follows the design elaborated in our paper: [TASP: Topology-aware Sequence Parallelism](https://arxiv.org/abs/2509.26541). Please refer to this paper if you are interested in the mathematical and CS-related principals behind the code in this repository.

**TLDR**: Hamilton Attention is a novel distributed attention mechanism that leverages Hamiltonian decomposition to optimize communication efficiency over Ring Attention. This pytorch implementation provides efficient distributed attention computation with support for both causal and non-causal attention patterns.

- In the first section (Hamilton Attention PyTorch Implementation), we present some overall structural and usage information of this pytorch implementation.
- In the second section (Hamilton Attention Test Script Usage), we present how to use the test script we provide to most easily reproduce our results.

## Table of Contents

- [Hamilton Attention](#hamilton-attention)
- [Hamilton Attention PyTorch Implementation](#hamilton-attention-pytorch-implementation)
  - [Communication Primitives](#communication-primitives)
  - [Distributed Attention API](#distributed-attention-api)
  - [Advanced Features](#advanced-features)
    - [Performance Profiling](#performance-profiling)
  - [Key Algorithms](#key-algorithms)
    - [Output and LSE Update](#output-and-lse-update)
    - [Cycle Generation Algorithms (Decomposition of Topology)](#cycle-generation-algorithms-decomposition-of-topology)
    - [Mapping Computation](#mapping-computation)
    - [Causal Masking](#causal-masking)
  - [Currently Supported Configurations](#currently-supported-configurations)
- [Hamilton Attention Test Script Usage](#hamilton-attention-test-script-usage)
  - [Test Script Usage](#test-script-usage)
    - [Basic Testing](#basic-testing)
    - [Example Test Configuration](#example-test-configuration)
  - [Test Outputs](#test-outputs)
    - [Performance Metrics](#performance-metrics)
    - [Accuracy Validation](#accuracy-validation)
  - [Advanced Testing Features](#advanced-testing-features)
    - [1. Profiling Modes](#1-profiling-modes)
    - [2. Trace Export](#2-trace-export)
  - [Running Tests](#running-tests)
    - [Single-node Test](#single-node-test)
    - [Multi-node Test](#multi-node-test)
  - [Configuration Parameters](#configuration-parameters)
    - [Model Dimensions](#model-dimensions)
    - [Test Configuration](#test-configuration)
    - [Advanced Options](#advanced-options)
- [Citation](#citation)


## Hamilton Attention PyTorch Implementation

### Communication Primitives

The `RingComm` class implements ring-based communication patterns, enabling efficient KV cache sharing between GPUs in a circular topology.
- Ring Communication (`RingComm`)
    - ```python
        class RingComm:
            def __init__(self, process_group: dist.ProcessGroup)
            def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.     Tensor] = None)
            def send_recv_kv(self, k: torch.Tensor, v: torch.Tensor,    k_buffer=None,     v_buffer=None)
      ```

The `A2AComm` class provides all-to-all collective communication, essential for the Hamilton path-based attention computation.
- All-to-All Communication (`A2AComm`)
    - ```python
      class A2AComm:
        def __init__(self, process_group: dist.ProcessGroup)
        def all_to_all(self, send_tensors: List[torch.Tensor], recv_tensors: List[torch.Tensor])
      ```



### Distributed Attention API

1. Standard Ring Attention
    - Implements the baseline ring attention algorithm with flash-attn integration.
    - Based on https://github.com/zhuzilin/ring-flash-attention
    - ```python
        def ring_attention_forward_flash_attn(
            process_group, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor,
            softmax_scale, 
            dropout_p=0, 
            causal=False, 
            window_size=(-1, -1)
        )
      ```
2. Zig-zag Ring Attention
    - Implements the baseline causal masking load-balanced ring attention algorithm with flash-attn integration.
    - Based on https://github.com/zhuzilin/ring-flash-attention
    - ```python
        def zigzag_ring_attention_forward_flash_attn(
            process_group,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            softmax_scale,
            dropout_p=0,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
      ```

3. $K_{m\times u}$-decomposition-based non-causal Hamilton Attention
    - Implements the $K_{m\times u}$-decomposition-based non-causal Hamilton Attention algorithm with flash-attn integration.
    - ```python
        def hamilton_attention_forward_full_flash_attn(
            process_group,
            q: torch.Tensor,                
            k: list[torch.Tensor],         
            v: list[torch.Tensor],         
            softmax_scale0,
            dropout_p=0,
            world_size=8,
            looping=None, 
            out_mapping=None,
            in_mapping=None, 
            layout="BSND"
        )
      ```
4. $K_{m\times u}$-decomposition-based causal Hamilton Attention
    - Implements the $K_{m\times u}$-decomposition-based causal Hamilton Attention algorithm with flash-attn integration.
    - ```python
        def hamilton_attention_forward_causal_flash_attn(
            process_group,
            q: torch.Tensor,        
            k: list[torch.Tensor],         
            v: list[torch.Tensor],         
            softmax_scale0,
            dropout_p=0,
            world_size=8,
            looping=None, 
            out_mapping=None,
            in_mapping=None, 
            source_rank_per_iter=None,
            layout="BSND",
        )
      ```
5. $(m-K_m-m)^u$-decomposition-based non-causal Hamilton Attention
    - Implements the $(m-K_m-m)^u$-decomposition-based non-causal Hamilton Attention algorithm with flash-attn integration.
    - ```python
        def hamilton_attention_forward_full_flash_attn_two_nodes(
            process_group,
            q: torch.Tensor,              
            k: list[torch.Tensor],          
            v: list[torch.Tensor],         
            softmax_scale0,
            dropout_p=0,
            inter_in_mapping=None,
            inter_out_mapping=None, 
            out_mapping=None,
            in_mapping=None, 
            layout="BSND"
        )
      ```




### Advanced Features

#### Performance Profiling

Profiling versions that return detailed timing information for communication and computation phases.

- The profiling versions require slightly more runtime overhead than non-profiling versions since it requires $2n$ cudaEvents/hipEvents ($n$ is the number of iterations, which is the same with #GPUs).
- ```python
    def ring_attention_forward_flash_attn_profile(...)
    def zigzag_ring_attention_forward_flash_attn_profile(...)
    def hamilton_attention_forward_full_flash_attn_profile(...)
    def hamilton_attention_forward_causal_flash_attn_profile(...)
    def hamilton_attention_forward_full_flash_attn_two_nodes_profile(...)
  ```

### Key Algorithms

#### Output and LSE Update

Efficiently combines partial attention results using numerically stable log-space operations.
- We utilize the implementation of https://github.com/zhuzilin/ring-flash-attention
    - ```python
       @torch.jit.script
       def _update_out_and_lse(
           out: torch.Tensor, lse: torch.Tensor, 
           block_out: torch.Tensor, block_lse: torch.Tensor
       )
       ```

#### Cycle Generation Algorithms (Decomposition of Topology)
Generates Hamiltonian cycles for complete graph communication patterns.
- Hamiltonian Cycle Generation for $K_{m\times u}$-decomposition ($n=m\times u$):
    - ```python
      def gen_hamilton_circle(n: int) -> List[List[int]]
      ```
- Hamiltonian Cycle Generation for $(m-K_m-m)^u$-decomposition ($m=8$):
    - ```python
        def gen_hamilton_circle_two_nodes(n: int) -> List[List[int]]
      ```

#### Mapping Computation
Compute routing tables of KV chunks for Hamiltonian cycles.

- For $K_{m\times u}$-decomposition ($n=m\times u$):
    - The inter-node and intra-node mappings are computed all-together, since we use AlltoAll. For AlltoAll, inter-node and intra-node communication are not distinguishable.
    - ```python
        def calculate_out_mapping_circle(looping: List[List[int]]) -> List[List[int]]
        def calculate_in_mapping_circle(looping: List[List[int]]) -> List[List[int]]  
        ```

- For $(m-K_m-m)^u$-decomposition ($m=8$):
    - The intra-node mapping are the same for each node. We can get node 1's mapping simply by adding $m$ to node 0's mapping.
        -   ```python
            def calculate_intra_node_out_mapping_two_nodes  (path0: List[List[int]]) ->   List[List[int]]
            def calculate_intra_node_in_mapping_two_nodes   (path0: List[List[int]]) ->    List[List[int]]
            ```
    - The inter-node mapping can be set manually or automatically generated (not implemented yet). Currently, we manually designate iter-node mapping for #node=2 and #node=4 while $m=8$.
        - ```python
            def calculate_inter_node_mapping(looping: List[List[int]]) -> List[(int,int)]
          ```
        - The first item of the tuple denotes the sender/receiver's global_rank (for in/out mapping), the second item denotes the KV chunk index.


#### Causal Masking

Compute the source rank of each received KV chunk at every iteration for every rank.
This is required when causal masking is used.

- Currently, we only support causal masking for $K_{m\times u}$-decomposition.
    - ```python
        def calculate_source_rank_per_iter(looping: List[List[int]])    -> List[List[List[bool]]]
      ```


### Currently Supported Configurations

#### For $K_{m\times u}$-decomposition ($n=m\times u$):
| GPU Count | #Node | Path Generation |
|-----------|----------|-----------------|
| 8 | Single Node | `gen_hamilton_circle(8)` |
| 16 | Two Nodes | `gen_hamilton_circle(16)` |
| 32 | Four Nodes | `gen_hamilton_circle(32)` |
| n ≥ 8, n % 4 == 0 | General | `gen_hamilton_circle(n)` |

#### For $(m-K_m-m)^u$-decomposition ($m=8$):
| GPU Count | #Node | Path Generation |
|-----------|----------|-----------------|
| 16 | Two Nodes | `gen_hamilton_path_local(8)` |
| 32 | Four Nodes | `gen_hamilton_path_local(8)` |
| u ≥ 1 | General | `gen_hamilton_path_local(8)` |


## Hamilton Attention Test Script Usage

### Test Script Usage

#### Basic Testing

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

#### Example Test Configuration

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


### Test Outputs

#### Performance Metrics

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

#### Accuracy Validation

When `accuracy_test=True`, the framework validates numerical correctness:

```python
# Cosine similarity between distributed and reference outputs
cosine_sim_out = F.cosine_similarity(
    torch.flatten(local_out), 
    torch.flatten(dist_out[iter0]), 
    dim=0
)
```

### Advanced Testing Features

#### 1. Profiling Modes

##### Performance Profiling
```python
# Enable detailed timing breakdown
profile_version = True
```

##### PyTorch Profiler Integration
```python
# Enable PyTorch profiler for detailed operator analysis
torch_profile_on = True
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
prof = torch.profiler.profile(activities=activities, schedule=schedule)
```

#### 2. Trace Export

Traces and prof_results are exported for every rank. The default root dir of trace and prof_results output are the dir of this test script itself.
```python
# Export Chrome trace for visualization in TensorBoard
prof.export_chrome_trace(f"{path_prefix}/trace_{config}_{local_rank}.json")
```

### Running Tests

#### Single-node Test

```bash
# Simple Node test
torchrun --nproc_per_node=8 test_hamilton_attention.py \
```

#### Multi-node Test

```bash
# 16-GPU two-node test  
torchrun --nnodes 2 --nproc_per_node=8 test_hamilton_attention.py \
```

### Configuration Parameters

#### Model Dimensions
| Parameter | Typical Values | Description |
|-----------|----------------|-------------|
| `bs` | 1-128 | Batch size |
| `seqlen` | 1K-1M | Sequence length (must be divisible by world_size) |
| `nheads` | 8-128 | Number of attention heads |
| `d` | 64, 128 | Head dimension |

**It should be noted**:
- **For $K_{m\times u}$-decomposition, seqlen must be divisible by world_size\*(world_size-1)**
- **For $(m-K_m-m)^u$-decomposition, seqlen must be divisible by world_size\*($m$)**

#### Test Configuration
| Parameter | Options | Description |
|-----------|---------|-------------|
| `causal` | True/False | Causal or non-causal attention |
| `two_nodes` | True/False | Use $(m-K_m-m)^u$-decomposition scheme |
| `test_nums` | 4-50 | Number of test iterations |
| `warm_ups` | 3-10 | Warm-up iterations before timing |

#### Advanced Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `accuracy_test` | False | Enable numerical accuracy validation |
| `profile_version` | True | Enable performance profiling |
| `torch_profile_on` | False | Enable PyTorch profiler |
| `log_all` | False | Enable detailed logging |

## Citation 

```tex
@misc{wang2025tasptopologyawaresequenceparallelism,
      title={TASP: Topology-aware Sequence Parallelism}, 
      author={Yida Wang and Ke Hong and Xiuhong Li and Yuanchao Xu and Wenxun Wang and Guohao Dai and Yu Wang},
      year={2025},
      eprint={2509.26541},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.26541}, 
}
```