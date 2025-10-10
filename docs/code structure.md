# Code Structure and API

In this markdown file we introduce the API and code structure of our pytorch implementation of Hamilton Attention. 

- [Code Structure and API](#code-structure-and-api)
  - [Hamilton Attention PyTorch Implementation](#hamilton-attention-pytorch-implementation)
    - [Communication Primitives](#communication-primitives)
    - [Distributed Attention API](#distributed-attention-api)
    - [Advanced Features](#advanced-features)
      - [Profiling-enabled API](#profiling-enabled-api)
    - [Key Algorithms](#key-algorithms)
      - [Output and LSE Update](#output-and-lse-update)
      - [Cycle Generation Algorithms (Decomposition of Topology)](#cycle-generation-algorithms-decomposition-of-topology)
      - [Mapping Computation](#mapping-computation)
      - [Causal Masking](#causal-masking)
    - [Currently Supported Configurations](#currently-supported-configurations)
      - [For $K\_{m\\times u}$-decomposition ($n=m\\times u$):](#for-k_mtimes-u-decomposition-nmtimes-u)
      - [For $(m-K\_m-m)^u$-decomposition ($m=8$):](#for-m-k_m-mu-decomposition-m8)


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

#### Profiling-enabled API

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
| GPU Count | #Node | Cycle Generation |
|-----------|----------|-----------------|
| 8 | Single Node | `gen_hamilton_circle(8)` |
| 16 | Two Nodes | `gen_hamilton_circle(16)` |
| 32 | Four Nodes | `gen_hamilton_circle(32)` |
| n ≥ 8, n % 4 == 0 | General | `gen_hamilton_circle(n)` |

#### For $(m-K_m-m)^u$-decomposition ($m=8$):
| GPU Count | #Node | Cycle Generation |
|-----------|----------|-----------------|
| 16 | Two Nodes | `gen_hamilton_path_local(8)` |
| 32 | Four Nodes | `gen_hamilton_path_local(8)` |
| u ≥ 1 | General | `gen_hamilton_path_local(8)` |

