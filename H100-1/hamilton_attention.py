import os

os.environ['TNG_LOG_LEVEL'] = '4'
os.environ['TORCH_LOGS'] = '+dynamo'
os.environ['TORCHDYNAMO_VERBOSE'] = '1'


from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
import torch.distributed as dist
import inspect
from functools import cache

from flash_attn.flash_attn_interface import _flash_attn_forward

from torch.cuda import Event

@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args

def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        return _get_default_args(func._init_fn)

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse

def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse

class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)
    
    def set_send_rank(self,send_rank):
        self.send_rank=send_rank

    def set_recv_rank(self,recv_rank):
        self.recv_rank=recv_rank

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res
    
    def send(
        self, send_rank: int,to_send: torch.Tensor
    ) -> torch.Tensor:


        send_op = dist.P2POp(
            dist.isend, to_send, send_rank, group=self._process_group
        )
        self._ops.append(send_op)

    def recv(
        self, recv_rank: int, recv_tensor: torch.Tensor 
    ) -> torch.Tensor:

        recv_op = dist.P2POp(dist.irecv, recv_tensor, recv_rank, group=self._process_group)
        self._ops.append(recv_op)
        return recv_tensor

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v

class A2AComm:
    def __init__(self,process_group: dist.ProcessGroup):
        self._process_group = process_group
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self.handles = []
    
    def all_to_all(
        self, send_tensors: List[torch.Tensor],recv_tensors: List[torch.Tensor],async_op=True
    ) -> torch.Tensor:
        assert send_tensors!=None and recv_tensors!=None 
        assert len(send_tensors)==len(recv_tensors)
        handle=dist.all_to_all(
            recv_tensors,send_tensors,group=self._process_group, async_op=async_op
        )
        self.handles.append(handle)
    
    def wait(self):
        if self.handles==None:
            raise RuntimeError("wait called before all_to_all!")
        for handle in self.handles:
            handle.wait()
        self.handles = []

def gen_hamilton_path_local(n):
    assert n==8, "currently n==8 is supported"
    return [
        [0,1,7,2,6,3,5,4],
        [1,2,0,3,7,4,6,5],
        [2,3,1,4,0,5,7,6],
        [3,4,2,5,1,6,0,7],
        [4,5,3,6,2,7,1,0],
        [5,6,4,7,3,0,2,1],
        [6,7,5,0,4,1,3,2],
        [7,0,6,1,5,2,4,3]
    ]

def gen_hamilton_circle_two_nodes(n):
    assert n==16, "currently n==16 is supported"
    res=[]
    p8=gen_hamilton_path_local(8)
    for row in p8:
        res.append(row+[x+8 for x in row])
    return res

def gen_hamilton_circle(n):
    if n==8:
        return [
            [7,0,1,5,2,4,3,6],
            [7,2,0,3,5,4,6,1],
            [6,2,3,1,4,0,7,5],
            [2,5,1,0,6,3,7,4],
            [2,1,6,4,5,7,3,0],
            [5,0,4,7,1,3,2,6],
            [5,3,4,1,2,7,6,0]
        ]
    
    assert n>=8, "n must be greater than or equal to 8"
    assert n%4==0, "n must be multiple of 4"

    init_looping=[]
    for i in range(n-2):
        if (i+1) % 2 ==1:
            init_looping.append((-(i//2))%(n-2))
        else:
            init_looping.append((i+1)//2)


    init_looping_2=[]
    for i in range(n-2):
        init_looping_2.append([n-2])
        for j in range(n-2):
            init_looping_2[-1].append((init_looping[j]+i)%(n-2))
    

    k=(n//4)-1  

    rotated=[]
    for i in range(n-2):
        shift=0
        if i==0:
            shift=1
        elif i==k+1:
            shift=4*k+2
        elif i==2*k+2:
            shift=3
        elif i==3*k+2:
            shift=4*k
        else:
            shift=2*k
        rotated.append([])
        for j in range(n-1):
            rotated[-1].append(init_looping_2[i][(j+shift)%(n-1)])


    appender={}
    for i in range(n-2):
        appender[rotated[i][-1]]=rotated[i][0]


    appender_list=[n-2]
    for i in range(n-2):
        appender_list.append(appender[appender_list[-1]])


    appendee=[]
    for i in range(n-2):
        appendee.append([n-1]+rotated[i])
    appendee.append([n-1]+appender_list)


    return appendee

def calculate_intra_node_out_mapping_two_nodes(path0):
    n=len(path0[0])
    m=len(path0)

    in_mapping=[]
    for local_rank in range(n):
        in_mapping.append([-1 for _ in range(n)])
        for row_idx in range(m):
            idx_of_local_rank=-1
            for i00 in range(n):
                if path0[row_idx][i00]==local_rank:
                    idx_of_local_rank=i00
                    break
            if idx_of_local_rank==n-1:
                continue
            else:
                in_mapping[-1][path0[row_idx][idx_of_local_rank+1]]=row_idx
    
    return in_mapping

def calculate_intra_node_in_mapping_two_nodes(path0):
    n=len(path0[0])
    m=len(path0)

    in_mapping=[]
    for local_rank in range(n):
        in_mapping.append([-1 for _ in range(n)])
        for row_idx in range(m):
            idx_of_local_rank=-1
            for i00 in range(n):
                if path0[row_idx][i00]==local_rank:
                    idx_of_local_rank=i00
                    break
            if idx_of_local_rank==0:
                continue
            else:
                in_mapping[-1][path0[row_idx][idx_of_local_rank-1]]=row_idx
    
    return in_mapping

def calculate_inter_node_mapping(looping):
    n = len(looping[0])
    m = len(looping)

    assert n==16 and m==8, "currently n==16 m==8 is supported"

    inter_mapping_in=[
        # 0~7
        (12,0),
        (13,1),
        (14,2),
        (15,3),
        (8,4),
        (9,5),
        (10,6),
        (11,7),
        # 8~15
        (4,0),
        (5,1),
        (6,2),
        (7,3),
        (0,4),
        (1,5),
        (2,6),
        (3,7),
    ]
    inter_mapping_out=[
        # 0~7
        (12,4),
        (13,5),
        (14,6),
        (15,7),
        (8,0),
        (9,1),
        (10,2),
        (11,3),
        # 8~15
        (4,4),
        (5,5),
        (6,6),
        (7,7),
        (0,0),
        (1,1),
        (2,2),
        (3,3),
    ]
    return inter_mapping_in,inter_mapping_out

def calculate_out_mapping_circle(looping):
    n = len(looping[0])
    m = len(looping)

    assert m==n-1, "m should be n-1 to form complete graph"

    mapping=[[-1] * n for _ in range(n)]
    for i in range(m):
        for j in range(n):
            mapping[looping[i][j]][looping[i][(j+1)%n]] = i 
    return mapping

def calculate_in_mapping_circle(looping):
    n = len(looping[0])
    m = len(looping)

    assert m==n-1, "m should be n-1 to form complete graph"

    mapping = [[-1] * n for _ in range(n)]
    for i in range(m):
        for j in range(n):
            mapping[looping[i][j]][looping[i][(j-1)%n]] = i 
    return mapping

def calculate_mask_index_circle(looping):
    n = len(looping[0])
    m = len(looping)

    mask = [[0] * m for _ in range(n)]
    for i in range(m):
        for j in range(n):
            mask[looping[i][j]][i] = j 
    return mask

def calculate_source_rank_per_iter(looping):
    n = len(looping[0])
    m = len(looping)

    res=[]
    for rank_id in range(n):
        res.append([])
        for iter_id in range(n):


            res[-1].append([])
            for chunk_id in range(m):
                rank_index_for_chunk = 0
                for (idx0,rank_idx0) in enumerate(looping[chunk_id]):
                    if rank_idx0==rank_id:
                        rank_index_for_chunk=idx0
                        break

                is_bigger = looping[chunk_id][(rank_index_for_chunk-iter_id)%n]>rank_id
                res[-1][-1].append(not is_bigger)
    
    return res

def get_circle_init_qkv_for_rank(looping,q_all,k_all,v_all,rank,layout="BSND"):
    n = len(looping[0])
    m = len(looping)

    assert layout in ["BSND","BNSD"], "layout must be BSND or BNSD"
    split_dim=1 if layout=="BSND" else 2
    q_chunks = q_all.chunk(n*m, dim=split_dim)
    k_chunks = k_all.chunk(n*m, dim=split_dim)
    v_chunks = v_all.chunk(n*m, dim=split_dim)
    local_q_list=[]
    local_k_list=[]
    local_v_list=[]
    for i in range(m):
        for j in range(n):
            if looping[i][j] == rank:
                local_q_list.append(q_chunks[i*n+j].detach().clone())
                local_k_list.append(k_chunks[i*n+j].detach().clone())
                local_v_list.append(v_chunks[i*n+j].detach().clone())
    local_q = torch.cat(local_q_list, dim=split_dim)
    return local_q, local_k_list, local_v_list

def get_circle_init_qkv_for_rank_dummy(looping,batch_size,seqlen,nheads,d,device,dtype,layout="BSND"):
    n = len(looping[0])
    m = len(looping)

    assert layout in ["BSND","BNSD"], "layout must be BSND or BNSD"
    split_dim=1 if layout=="BSND" else 2

    assert seqlen % (m*n)==0
    local_q=torch.randn(batch_size, seqlen//n, nheads, d, device=device, dtype=dtype)
    local_v=torch.randn(batch_size, seqlen//n, nheads, d, device=device, dtype=dtype)
    local_k=torch.randn(batch_size, seqlen//n, nheads, d, device=device, dtype=dtype)
    
    local_k_list=local_k.chunk(m,dim=split_dim)
    local_v_list=local_v.chunk(m,dim=split_dim)

    return local_q, local_k_list, local_v_list

def extract_local_for_ring_dummy(world_size,batch_size,seqlen,nheads,d,device,dtype, dim=1):
    return torch.randn(batch_size, seqlen//world_size, nheads, d, device=device, dtype=dtype)

def get_circle_local_out_lse_for_rank(looping,out,lse0,rank, layout="BSND"):
    n= len(looping[0])
    m= len(looping)
    assert layout in ["BSND", "BNSD"], "layout must be BSND or BNSD"


    out_chunks = out.chunk(n*m, dim=1 if layout=="BSND" else 2)
    lse_chunks = lse0.chunk(n*m, dim= 2)
    local_out_list=[]
    local_lse_list=[]

    for i in range(m):
        for j in range(n):
            if looping[i][j] == rank:
                local_out_list.append(out_chunks[i*n+j].detach().clone())
                local_lse_list.append(lse_chunks[i*n+j].detach().clone())
    local_out = torch.cat(local_out_list, dim=1 if layout=="BSND" else 2)
    local_lse = torch.cat(local_lse_list, dim=2)
    return local_out, local_lse

def get_circle_init_qkv_for_rank_causal(looping,q_all,k_all,v_all,rank,layout="BSND"):
    n = len(looping[0])
    m = len(looping)


    assert layout in ["BSND","BNSD"], "layout must be BSND or BNSD"
    split_dim=1 if layout=="BSND" else 2
    q_chunks = q_all.chunk(n*2*m, dim=split_dim)
    k_chunks = k_all.chunk(n*2*m, dim=split_dim)
    v_chunks = v_all.chunk(n*2*m, dim=split_dim)
    local_q_list=[]
    local_k_list=[]
    local_v_list=[]
    i = rank
    for j in range(m):
        local_q_list.append(q_chunks[i*m+j].detach().clone())
        
        local_k_0=torch.cat([k_chunks[i*m+j].detach().clone(),k_chunks[2*m*n-m-i*m+j].detach().clone()],dim=split_dim)
        local_k_list.append(local_k_0)
        
        local_v_0=torch.cat([v_chunks[i*m+j].detach().clone(),v_chunks[2*m*n-m-i*m+j].detach().clone()],dim=split_dim)
        local_v_list.append(local_v_0)
    for j in range(m):
        local_q_list.append(q_chunks[2*m*n-m-i*m+j].detach().clone())
    local_q = torch.cat(local_q_list, dim=split_dim)
    return local_q, local_k_list, local_v_list

def get_circle_local_out_lse_for_rank_causal(looping,out,lse0,rank, layout="BSND"):
    n= len(looping[0])
    m= len(looping)
    assert layout in ["BSND", "BNSD"], "layout must be BSND or BNSD"


    out_chunks = out.chunk(2*n*m, dim=1 if layout=="BSND" else 2)
    lse_chunks = lse0.chunk(2*n*m, dim= 2)
    local_out_list=[]
    local_lse_list=[]

    i = rank
    for j in range(m):
        local_out_list.append(out_chunks[i*m+j].detach().clone())
        local_lse_list.append(lse_chunks[i*m+j].detach().clone())
    for j in range(m):
        local_out_list.append(out_chunks[2*m*n-m-i*m+j].detach().clone())
        local_lse_list.append(lse_chunks[2*m*n-m-i*m+j].detach().clone())
    local_out = torch.cat(local_out_list, dim=1 if layout=="BSND" else 2)
    local_lse = torch.cat(local_lse_list, dim=2)
    return local_out, local_lse

def extract_local_for_zigzag_ring(x_all, rank, world_size, dim=1):
    x_chunks = x_all.chunk(2 * world_size, dim=dim)

    local_x = torch.cat(
        [x_chunks[rank], x_chunks[2 * world_size - rank - 1]], dim=dim
    ).contiguous()
    
    return local_x

def extract_local_for_ring(x_all, rank, world_size, dim=1):
    x_chunks = x_all.chunk(  world_size, dim=dim)    
    return x_chunks[rank].detach().clone()

def ring_attention_forward_flash_attn(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        if not causal or step <= comm.rank:
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal and step == 0,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                }
            )
            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "window_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )
            outputs = _flash_attn_forward(**params)
            if len(outputs) == 8:
                block_out, _, _, _, _, block_lse, _, _ = outputs
            else:
                assert len(outputs) == 4
                block_out, block_lse, _, _ = outputs
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

def ring_attention_forward_flash_attn_profile(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    world_size=8,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    comm = RingComm(process_group)
    stream_comm=torch.cuda.Stream()

    out = None
    lse = None

    next_k, next_v = None, None

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    #### profile part start ####
    total_start_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)
    
    step_events = []
    for step in range(world_size):
        step_events.append({
            'comm_start': torch.cuda.Event(enable_timing=True),
            'comm_end': torch.cuda.Event(enable_timing=True),
            'comp_start': torch.cuda.Event(enable_timing=True),
            'comp_end': torch.cuda.Event(enable_timing=True),
        })
    #### profile part end ####
    
    #### profile part start ####
    total_start_event.record()
    #### profile part end ####


    for step in range(comm.world_size):
        with torch.cuda.stream(stream_comm):
            #### profile part start ####
            step_events[step]['comm_start'].record()
            #### profile part end ####
            if step + 1 != comm.world_size:
                next_k, next_v = comm.send_recv_kv(k, v)
                comm.wait()
            #### profile part start ####
            step_events[step]['comm_end'].record()
            #### profile part end ####
        
        #### profile part start ####
        step_events[step]['comp_start'].record()
        #### profile part end ####
        
        if not causal or step <= comm.rank:
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal and step == 0,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                }
            )
            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "window_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )
            outputs = _flash_attn_forward(**params)
            if len(outputs) == 8:
                block_out, _, _, _, _, block_lse, _, _ = outputs
            else:
                assert len(outputs) == 4
                block_out, block_lse, _, _ = outputs
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        
        #### profile part start ####
        step_events[step]['comp_end'].record()
        #### profile part end ####
        
        if step + 1 != comm.world_size:
            # comm.wait()
            stream_comm.synchronize()
            k, v = next_k, next_v
    
    
    #### profile part start ####
    total_end_event.record()
    torch.cuda.synchronize()

    total_time = total_start_event.elapsed_time(total_end_event) 
    comm_total_time = 0.0
    comp_total_time = 0.0
    for step in range(world_size):
        comm_time = step_events[step]['comm_start'].elapsed_time(step_events[step]['comm_end'])
        comm_total_time += comm_time
        
        comp_time = step_events[step]['comp_start'].elapsed_time(step_events[step]['comp_end'])
        comp_total_time += comp_time
    #### profile part end ####

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse,{
        'total_time': total_time,
        'comm_total_time': comm_total_time,
        'comp_total_time': comp_total_time
    }

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
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]

    out = None
    lse = None
    next_k, next_v = None, None

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    def forward(q, k, v, causal):
        params = get_default_args(_flash_attn_forward).copy()
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
        outputs = _flash_attn_forward(**params)
        if len(outputs) == 8:
            block_out, _, _, _, _, block_lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs
        return block_out, block_lse

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        if step == 0:
            block_out, block_lse = forward(q, k, v, causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_lse = forward(q, k0, v0, causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, causal=False)
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

def zigzag_ring_attention_forward_flash_attn_profile(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    world_size=8,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert causal == True, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)
    stream_comm=torch.cuda.Stream()

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]

    out = None
    lse = None
    next_k, next_v = None, None

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    #### profile part start ####
    total_start_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)
    
    step_events = []
    for step in range(world_size):
        step_events.append({
            'comm_start': torch.cuda.Event(enable_timing=True),
            'comm_end': torch.cuda.Event(enable_timing=True),
            'comp_start': torch.cuda.Event(enable_timing=True),
            'comp_end': torch.cuda.Event(enable_timing=True),
        })
    #### profile part end ####

    def forward(q, k, v, causal):
        params = get_default_args(_flash_attn_forward).copy()
        params.update(
            {
                "q": q,
                "k": k,
                "v": v,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "alibi_slopes": alibi_slopes,
                "return_softmax": True and dropout_p > 0,
            }
        )
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
        outputs = _flash_attn_forward(**params)
        if len(outputs) == 8:
            block_out, _, _, _, _, block_lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs
        return block_out, block_lse

    #### profile part start ####
    total_start_event.record()
    #### profile part end ####

    for step in range(comm.world_size):
        
        with torch.cuda.stream(stream_comm):
            #### profile part start ####
            step_events[step]['comm_start'].record()
            #### profile part end ####
            if step + 1 != comm.world_size:
                next_k, next_v = comm.send_recv_kv(k, v)
                comm.wait()
            #### profile part start ####
            step_events[step]['comm_end'].record()
            #### profile part end ####

        #### profile part start ####
        step_events[step]['comp_start'].record()
        #### profile part end ####
        if step == 0:
            block_out, block_lse = forward(q, k, v, causal=True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[:, :block_seq_len]
            v0 = v[:, :block_seq_len]
            block_out, block_lse = forward(q, k0, v0, causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, causal=False)
            out, lse = update_out_and_lse(
                out,
                lse,
                block_out,
                block_lse,
                slice_=(slice(None), slice(block_seq_len, None)),
            )
        #### profile part start ####
        step_events[step]['comp_end'].record()
        #### profile part end ####

        if step + 1 != comm.world_size:
            # comm.wait()
            stream_comm.synchronize()
            k, v = next_k, next_v

    #### profile part start ####
    total_end_event.record()
    torch.cuda.synchronize()

    total_time = total_start_event.elapsed_time(total_end_event)  
    comm_total_time = 0.0
    comp_total_time = 0.0
    for step in range(world_size):
        comm_time = step_events[step]['comm_start'].elapsed_time(step_events[step]['comm_end'])
        comm_total_time += comm_time
        
        comp_time = step_events[step]['comp_start'].elapsed_time(step_events[step]['comp_end'])
        comp_total_time += comp_time
    #### profile part end ####

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse,{
        'total_time': total_time,
        'comm_total_time': comm_total_time,
        'comp_total_time': comp_total_time
    }

def hamilton_attention_forward_full_flash_attn(
    process_group,
    q: torch.Tensor,                # (batch_size, seqlen, nheads, headdim) BSND /  (batch_size, nheads, seqlen, headdim) BNSD 
    k: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    v: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    softmax_scale0,
    dropout_p=0,
    world_size=8,
    looping=None, 
    out_mapping=None,
    in_mapping=None, 
    layout="BSND"
):
    assert layout in ["BSND", "BNSD"], "layout must be BSND or BNSD"
    if looping is None:
        looping=gen_hamilton_circle(world_size)
    if out_mapping is None:
        out_mapping = calculate_out_mapping_circle(looping)
    if in_mapping is None:
        in_mapping = calculate_in_mapping_circle(looping)

    if softmax_scale0 is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    comm = A2AComm(process_group)
    current_rank= comm.rank

    out_mapping_current_rank = out_mapping[current_rank]
    in_mapping_current_rank = in_mapping[current_rank]


    out = None
    lse=None

    para_size=world_size-1

    this_kv=[torch.stack((k[i],v[i]), dim=0) for i in range(para_size)]
    k, v= None, None
    next_kv= [torch.empty_like(this_kv[0]) for _ in range(para_size)]

    
    def forward(q, k, v):
        outs = _flash_attn_forward(
            q=q,
            k=k,
            v=v,
            dropout_p=0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False
        )
        if len(outs)==8:
            block_out, _, _, _, _, block_softmax_lse, _, _, = outs
        elif len(outs)==4:
            block_out, block_softmax_lse, _, _, = outs

        return block_out, block_softmax_lse


    dim_seqlen=1 if layout=="BSND" else 2
    for step in range(comm.world_size):

        k= [this_kv[i][0] for i in range(para_size)]
        v= [this_kv[i][1] for i in range(para_size)]

        if step + 1 != comm.world_size:

            kv_send=[
                 this_kv[out_mapping_current_rank[i]] \
                    if out_mapping_current_rank[i]!=-1 else torch.empty(0, dtype=q.dtype, device=q.device) \
                        for i in range(world_size)]
            kv_recv=[
                    next_kv[in_mapping_current_rank[i]] \
                    if in_mapping_current_rank[i]!=-1 else torch.empty(0, dtype=q.dtype, device=q.device) \
                        for i in range(world_size)]
            comm.all_to_all(kv_send,kv_recv)

        k_concat = torch.cat(k, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
        v_concat = torch.cat(v, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
        block_out, block_lse = forward(q, k_concat, v_concat)
        out, lse = update_out_and_lse(out, lse,block_out, block_lse)


        if step + 1 != comm.world_size:
            comm.wait()
            tmp=this_kv
            this_kv=next_kv
            next_kv=tmp

    
    out = out.to(q.dtype)
    return out, lse

def hamilton_attention_forward_full_flash_attn_two_nodes(
    process_group,
    q: torch.Tensor,                # (batch_size, seqlen, nheads, headdim) BSND /  (batch_size, nheads, seqlen, headdim) BNSD 
    k: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    v: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    softmax_scale0,
    dropout_p=0,
    inter_in_mapping=None,
    inter_out_mapping=None, 
    out_mapping=None,
    in_mapping=None, 
    layout="BSND"
):
    assert layout in ["BSND", "BNSD"], "layout must be BSND or BNSD"

    if softmax_scale0 is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    
    # 8 ranks per node
    local_world_size=8
    # 16 ranks 
    global_world_size=16



    # ring comm on the global rank
    ring_comm = RingComm(process_group)
    current_rank= ring_comm.rank



    send_rank=inter_out_mapping[current_rank][0]
    send_chunk_id=inter_out_mapping[current_rank][1]
    
    recv_rank=inter_in_mapping[current_rank][0]
    recv_chunk_id=inter_in_mapping[current_rank][1]

    ring_comm.set_send_rank(send_rank)
    ring_comm.set_recv_rank(recv_rank)


    node_rank = current_rank //  local_world_size
    


    stream_comm=torch.cuda.Stream()

    out_mapping_current_rank = out_mapping[current_rank%local_world_size]
    in_mapping_current_rank = in_mapping[current_rank%local_world_size]

    out = None
    lse=None

    chunk_num_per_rank=8

    this_kv=[torch.stack((k[i],v[i]), dim=0) for i in range(chunk_num_per_rank)]
    k, v= None, None
    next_kv= [torch.empty_like(this_kv[0]) for _ in range(chunk_num_per_rank)]

    
    def forward(q, k, v):
        outs = _flash_attn_forward(
            q=q,
            k=k,
            v=v,
            dropout_p=0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False
        )
        if len(outs)==8:
            block_out, _, _, _, _, block_softmax_lse, _, _, = outs
        elif len(outs)==4:
            block_out, block_softmax_lse, _, _, = outs

        return block_out, block_softmax_lse


    dim_seqlen=1 if layout=="BSND" else 2
    for step in range(global_world_size):
        

        k= [this_kv[i][0] for i in range(chunk_num_per_rank)]
        v= [this_kv[i][1] for i in range(chunk_num_per_rank)]

        with torch.cuda.stream(stream_comm):

            if step + 1 != global_world_size:
                next_kv[recv_chunk_id]=ring_comm.send_recv(this_kv[send_chunk_id],next_kv[recv_chunk_id])



                for rank0 in range(local_world_size):
                    if out_mapping_current_rank[rank0]!=-1:
                        ring_comm.send(
                            node_rank*local_world_size+rank0,this_kv[out_mapping_current_rank[rank0]]
                        )
                    if in_mapping_current_rank[rank0]!=-1:
                        ring_comm.recv(
                            node_rank*local_world_size+rank0,next_kv[in_mapping_current_rank[rank0]]
                        )

                ring_comm.commit()
                ring_comm.wait()



        k_concat = torch.cat(k, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
        v_concat = torch.cat(v, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
        block_out, block_lse = forward(q, k_concat, v_concat)
        out, lse = update_out_and_lse(out, lse,block_out, block_lse)



        if step + 1 != global_world_size:
            stream_comm.synchronize()

            tmp=this_kv
            this_kv=next_kv
            next_kv=tmp
   
    
    out = out.to(q.dtype)
    return out, lse

def hamilton_attention_forward_full_flash_attn_two_nodes_profile(
    process_group,
    q: torch.Tensor,                # (batch_size, seqlen, nheads, headdim) BSND /  (batch_size, nheads, seqlen, headdim) BNSD 
    k: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    v: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    softmax_scale0,
    dropout_p=0,
    inter_in_mapping=None,
    inter_out_mapping=None, 
    out_mapping=None,
    in_mapping=None, 
    layout="BSND"
):
    assert layout in ["BSND", "BNSD"], "layout must be BSND or BNSD"

    if softmax_scale0 is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    
    # 8 ranks per node
    local_world_size=8
    # 16 ranks 
    global_world_size=16



    # ring comm on the global rank
    ring_comm = RingComm(process_group)
    stream_ring_comm=torch.cuda.Stream()

    current_rank= ring_comm.rank


    send_rank=inter_out_mapping[current_rank][0]
    send_chunk_id=inter_out_mapping[current_rank][1]
    
    recv_rank=inter_in_mapping[current_rank][0]
    recv_chunk_id=inter_in_mapping[current_rank][1]

    ring_comm.set_send_rank(send_rank)
    ring_comm.set_recv_rank(recv_rank)


    node_rank = current_rank //  local_world_size
    
    local_ranks = list(range(node_rank * local_world_size, 
                           (node_rank + 1) * local_world_size))
    

    stream_comm=torch.cuda.Stream()

    out_mapping_current_rank = out_mapping[current_rank%local_world_size]
    in_mapping_current_rank = in_mapping[current_rank%local_world_size]

    out = None
    lse=None

    chunk_num_per_rank=8

    this_kv=[torch.stack((k[i],v[i]), dim=0) for i in range(chunk_num_per_rank)]
    k, v= None, None
    next_kv= [torch.empty_like(this_kv[0]) for _ in range(chunk_num_per_rank)]

    #### profile part start ####
    total_start_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)
    
    step_events = []
    for step in range(global_world_size):
        step_events.append({
            'comm_start': torch.cuda.Event(enable_timing=True),
            'comm_end': torch.cuda.Event(enable_timing=True),
            'comp_start': torch.cuda.Event(enable_timing=True),
            'comp_end': torch.cuda.Event(enable_timing=True),
        })
    #### profile part end ####
    
    def forward(q, k, v):
        outs = _flash_attn_forward(
            q=q,
            k=k,
            v=v,
            dropout_p=0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False
        )
        if len(outs)==8:
            block_out, _, _, _, _, block_softmax_lse, _, _, = outs
        elif len(outs)==4:
            block_out, block_softmax_lse, _, _, = outs

        return block_out, block_softmax_lse

    #### profile part start ####
    total_start_event.record()
    #### profile part end ####

    dim_seqlen=1 if layout=="BSND" else 2
    for step in range(global_world_size):
        
        # if current_rank==0:
        #     print(f"begin iter {step} ")

        k= [this_kv[i][0] for i in range(chunk_num_per_rank)]
        v= [this_kv[i][1] for i in range(chunk_num_per_rank)]

        with torch.cuda.stream(stream_comm):

            #### profile part start ####
            step_events[step]['comm_start'].record()
            #### profile part end ####
            if step + 1 != global_world_size:
                next_kv[recv_chunk_id]=ring_comm.send_recv(this_kv[send_chunk_id],next_kv[recv_chunk_id])



                for rank0 in range(local_world_size):
                    if out_mapping_current_rank[rank0]!=-1:
                        ring_comm.send(
                            node_rank*local_world_size+rank0,this_kv[out_mapping_current_rank[rank0]]
                        )
                    if in_mapping_current_rank[rank0]!=-1:
                        ring_comm.recv(
                            node_rank*local_world_size+rank0,next_kv[in_mapping_current_rank[rank0]]
                        )

                ring_comm.commit()
                ring_comm.wait()

            #### profile part start ####
            step_events[step]['comm_end'].record()
            #### profile part end ####



        #### profile part start ####
        step_events[step]['comp_start'].record()
        #### profile part end ####
        k_concat = torch.cat(k, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
        v_concat = torch.cat(v, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
        block_out, block_lse = forward(q, k_concat, v_concat)
        out, lse = update_out_and_lse(out, lse,block_out, block_lse)

        #### profile part start ####
        step_events[step]['comp_end'].record()
        #### profile part end ####


        if step + 1 != global_world_size:
            stream_comm.synchronize()

            tmp=this_kv
            this_kv=next_kv
            next_kv=tmp
   
    #### profile part start ####
    total_end_event.record()
    torch.cuda.synchronize()

    total_time = total_start_event.elapsed_time(total_end_event)  
    comm_total_time = 0.0
    comp_total_time = 0.0
    # ring_comm_total_time=0.0
    for step in range(global_world_size):
        comm_time = step_events[step]['comm_start'].elapsed_time(step_events[step]['comm_end'])
        comm_total_time += comm_time

        
        comp_time = step_events[step]['comp_start'].elapsed_time(step_events[step]['comp_end'])
        comp_total_time += comp_time
    #### profile part end ####
    
    out = out.to(q.dtype)
    return out, lse,{
        'total_time': total_time,
        'comm_total_time': comm_total_time,
        # 'ring_comm_total_time': ring_comm_total_time,
        'comp_total_time': comp_total_time
    }

def hamilton_attention_forward_full_flash_attn_profile(
    process_group,
    q: torch.Tensor,                # (batch_size, seqlen, nheads, headdim) BSND /  (batch_size, nheads, seqlen, headdim) BNSD 
    k: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    v: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    softmax_scale0,
    dropout_p=0,
    world_size=8,
    looping=None, 
    out_mapping=None,
    in_mapping=None, 
    layout="BSND"
):
    assert layout in ["BSND", "BNSD"], "layout must be BSND or BNSD"
    if looping is None:
        looping=gen_hamilton_circle(world_size)
    if out_mapping is None:
        out_mapping = calculate_out_mapping_circle(looping)
    if in_mapping is None:
        in_mapping = calculate_in_mapping_circle(looping)

    if softmax_scale0 is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    comm = A2AComm(process_group)
    current_rank= comm.rank

    out_mapping_current_rank = out_mapping[current_rank]
    in_mapping_current_rank = in_mapping[current_rank]

    stream_comm=torch.cuda.Stream()


    out = None
    lse=None

    para_size=world_size-1

    this_kv=[torch.stack((k[i],v[i]), dim=0) for i in range(para_size)]
    k, v= None, None
    next_kv= [torch.empty_like(this_kv[0]) for _ in range(para_size)]

    #### profile part start ####
    total_start_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)
    step_events = []
    for step in range(world_size):
        step_events.append({
            'comm_start': torch.cuda.Event(enable_timing=True),
            'comm_end': torch.cuda.Event(enable_timing=True),
            'comp_start': torch.cuda.Event(enable_timing=True),
            'comp_end': torch.cuda.Event(enable_timing=True),
        })
    #### profile part end ####
    
    def forward(q, k, v):
        outs = _flash_attn_forward(
            q=q,
            k=k,
            v=v,
            dropout_p=0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False
        )
        if len(outs)==8:
            block_out, _, _, _, _, block_softmax_lse, _, _, = outs
        elif len(outs)==4:
            block_out, block_softmax_lse, _, _, = outs
        return block_out, block_softmax_lse


    dim_seqlen=1 if layout=="BSND" else 2

    #### profile part start ####
    total_start_event.record()
    #### profile part end ####


    for step in range(comm.world_size):

        k= [this_kv[i][0] for i in range(para_size)]
        v= [this_kv[i][1] for i in range(para_size)]

        with torch.cuda.stream(stream_comm):

            #### profile part start ####
            step_events[step]['comm_start'].record()
            #### profile part end ####

            if step + 1 != comm.world_size:

                kv_send=[
                     this_kv[out_mapping_current_rank[i]] \
                        if out_mapping_current_rank[i]!=-1 else torch.empty(0, dtype=q. dtype, device=q.device) \
                            for i in range(world_size)]
                kv_recv=[
                        next_kv[in_mapping_current_rank[i]] \
                        if in_mapping_current_rank[i]!=-1 else torch.empty(0, dtype=q.  dtype, device=q.device) \
                            for i in range(world_size)]
                comm.all_to_all(kv_send,kv_recv)
                comm.wait()

            #### profile part start ####
            step_events[step]['comm_end'].record()
            #### profile part end ####
        
        #### profile part start ####
        step_events[step]['comp_start'].record()
        #### profile part end ####


        k_concat = torch.cat(k, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
        v_concat = torch.cat(v, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
        block_out, block_lse = forward(q, k_concat, v_concat)
        out, lse = update_out_and_lse(out, lse,block_out, block_lse)

        #### profile part start ####
        step_events[step]['comp_end'].record()
        #### profile part end ####

        if step + 1 != comm.world_size:
            # comm.wait()
            stream_comm.synchronize()
            tmp=this_kv
            this_kv=next_kv
            next_kv=tmp
    
    #### profile part start ####
    total_end_event.record()
    torch.cuda.synchronize()

    total_time = total_start_event.elapsed_time(total_end_event)  
    comm_total_time = 0.0
    comp_total_time = 0.0
    for step in range(world_size):
        comm_time = step_events[step]['comm_start'].elapsed_time(step_events[step]['comm_end'])
        comm_total_time += comm_time
        
        comp_time = step_events[step]['comp_start'].elapsed_time(step_events[step]['comp_end'])
        comp_total_time += comp_time
    #### profile part end ####


    out = out.to(q.dtype)
    return out, lse,{
        'total_time': total_time,
        'comm_total_time': comm_total_time,
        'comp_total_time': comp_total_time
    }

def hamilton_attention_forward_causal_flash_attn(
    process_group,
    q: torch.Tensor,                # (batch_size, seqlen, nheads, headdim) BSND /  (batch_size, nheads, seqlen, headdim) BNSD 
    k: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    v: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    softmax_scale0,
    dropout_p=0,
    world_size=8,
    looping=None, 
    out_mapping=None,
    in_mapping=None, 
    source_rank_per_iter=None,
    layout="BSND",
):

    assert layout in ["BSND", "BNSD"], "layout must be BSND or BNSD"
    # if looping not provided, generate one
    if looping is None:
        looping=gen_hamilton_circle(world_size)
    if out_mapping is None:
        out_mapping = calculate_out_mapping_circle(looping)
    if in_mapping is None:
        in_mapping = calculate_in_mapping_circle(looping)
    if source_rank_per_iter is None:
        source_rank_per_iter = calculate_source_rank_per_iter(looping)

    if softmax_scale0 is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q1=q.chunk(2, dim=1)[1]
    block_seq_len = q.shape[1] // 2

    comm = A2AComm(process_group)
    current_rank= comm.rank

    out_mapping_current_rank = out_mapping[current_rank]
    in_mapping_current_rank = in_mapping[current_rank]
    source_rank_per_iter_current_rank = source_rank_per_iter[current_rank]

    out = None
    lse=None

    para_size=world_size-1

    this_kv=[torch.stack((k[i],v[i]), dim=0) for i in range(para_size)]
    k, v= None, None
    next_kv= [torch.empty_like(this_kv[0]) for _ in range(para_size)]

    def forward(q, k, v,causal):
        outs = _flash_attn_forward(
            q=q,
            k=k,
            v=v,
            dropout_p=0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False
        )
        if len(outs)==8:
            block_out, _, _, _, _, block_softmax_lse, _, _, = outs
        elif len(outs)==4:
            block_out, block_softmax_lse, _, _, = outs

        return block_out, block_softmax_lse
    
    dim_seqlen=1 if layout=="BSND" else 2
    
    # pre-allocate buffer
    Mk_concat_buffer = None
    Mv_concat_buffer = None
    Pk_concat_buffer = None
    Pv_concat_buffer = None

    
    for step in range(comm.world_size):

        k= [this_kv[i][0] for i in range(para_size)]
        v= [this_kv[i][1] for i in range(para_size)]

        if step + 1 != comm.world_size:
            # fetch KV chunks from other ranks

            kv_send=[
                 this_kv[out_mapping_current_rank[i]] \
                    if out_mapping_current_rank[i]!=-1 else torch.empty(0, dtype=q.dtype, device=q.device) \
                        for i in range(world_size)]
            kv_recv=[
                    next_kv[in_mapping_current_rank[i]] \
                    if in_mapping_current_rank[i]!=-1 else torch.empty(0, dtype=q.dtype, device=q.device) \
                        for i in range(world_size)]
            comm.all_to_all(kv_send,kv_recv)

        if step == 0 :
            # causal mask
            k_split=[None]*(2*para_size)
            v_split=[None]*(2*para_size)
            for idx00,(k_slice,v_slice) in enumerate(zip(k,v)):
                
                k_chunks0=k_slice.chunk(2,dim=dim_seqlen)
                k_split[idx00]=k_chunks0[0]
                k_split[idx00+para_size]=k_chunks0[1]

                v_chunks0=v_slice.chunk(2,dim=dim_seqlen)
                v_split[idx00]=v_chunks0[0]
                v_split[idx00+para_size]=v_chunks0[1]
            k_concat = torch.cat(k_split, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
            v_concat = torch.cat(v_split, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
            block_out, block_lse = forward(q, k_concat, v_concat,causal=True)
            out, lse = update_out_and_lse(out, lse,block_out, block_lse)
        else:
            assert step<len(source_rank_per_iter_current_rank), f"step {step} is list index out of range {len(source_rank_per_iter_current_rank)}"
            source_rank= source_rank_per_iter_current_rank[step]
            Mk_list=[]
            Mv_list=[]
            Pk_list=[]
            Pv_list=[]
            # query can be divided into two parts A, B
            # received KV chunks can be divided into 2(n-1) parts, each one of the n-1 parts can be divided into two parts M, N
            for idx01 in range(para_size):
                # M are passed from source ranks that have smaller rank id than current rank 
                if source_rank[idx01]:
                    Mk_list.append(k[idx01].chunk(2,dim=1)[0])
                    Mv_list.append(v[idx01].chunk(2,dim=1)[0])
                else:
                    Pk_list.append(k[idx01])
                    Pv_list.append(v[idx01])
            
            # use pre-allocated buffers
            if len(Mk_list)>0:
                # reuse or re-allocate buffers
                if Mk_concat_buffer is None or Mk_concat_buffer.shape[dim_seqlen] != sum(m.shape[dim_seqlen] for m in Mk_list):
                    Mk_concat = torch.cat(Mk_list, dim=dim_seqlen)
                    Mv_concat = torch.cat(Mv_list, dim=dim_seqlen)
                    # renew buffers
                    Mk_concat_buffer = Mk_concat
                    Mv_concat_buffer = Mv_concat
                else:
                    # reuse buffers
                    start_idx = 0
                    for i, m in enumerate(Mk_list):
                        end_idx = start_idx + m.shape[dim_seqlen]
                        Mk_concat_buffer.narrow(dim_seqlen, start_idx, m.shape[dim_seqlen]).copy_(m)
                        Mv_concat_buffer.narrow(dim_seqlen, start_idx, Mv_list[i].shape[dim_seqlen]).copy_(Mv_list[i])
                        start_idx = end_idx
                    
                block_out, block_lse = forward(q, Mk_concat_buffer, Mv_concat_buffer, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            
            if len(Pk_list)>0:
                # reuse or re-allocate buffers
                if Pk_concat_buffer is None or Pk_concat_buffer.shape[dim_seqlen] != sum(p.shape[dim_seqlen] for p in Pk_list):
                    Pk_concat = torch.cat(Pk_list, dim=dim_seqlen)
                    Pv_concat = torch.cat(Pv_list, dim=dim_seqlen)
                    # renew buffers
                    Pk_concat_buffer = Pk_concat
                    Pv_concat_buffer = Pv_concat
                else:
                    # reuse buffers
                    start_idx = 0
                    for i, p in enumerate(Pk_list):
                        end_idx = start_idx + p.shape[dim_seqlen]
                        Pk_concat_buffer.narrow(dim_seqlen, start_idx, p.shape[dim_seqlen]).copy_(p)
                        Pv_concat_buffer.narrow(dim_seqlen, start_idx, Pv_list[i].shape[dim_seqlen]).copy_(Pv_list[i])
                        start_idx = end_idx
                
                block_out1, block_lse1 = forward(q1, Pk_concat_buffer, Pv_concat_buffer, causal=False)
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out1,
                    block_lse1,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )
            

        if step + 1 != comm.world_size:
            comm.wait()
            tmp=this_kv
            this_kv=next_kv
            next_kv=tmp

    
    out = out.to(q.dtype)
    return out, lse

def hamilton_attention_forward_causal_flash_attn_profile(
    process_group,
    q: torch.Tensor,                # (batch_size, seqlen, nheads, headdim) BSND /  (batch_size, nheads, seqlen, headdim) BNSD 
    k: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    v: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    softmax_scale0,
    dropout_p=0,
    world_size=8,
    looping=None, 
    out_mapping=None,
    in_mapping=None, 
    source_rank_per_iter=None,
    layout="BSND",
):
    assert layout in ["BSND", "BNSD"], "layout must be BSND or BNSD"
    # if looping not provided, generate one
    if looping is None:
        looping=gen_hamilton_circle(world_size)
    if out_mapping is None:
        out_mapping = calculate_out_mapping_circle(looping)
    if in_mapping is None:
        in_mapping = calculate_in_mapping_circle(looping)
    if source_rank_per_iter is None:
        source_rank_per_iter = calculate_source_rank_per_iter(looping)

    if softmax_scale0 is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q1=q.chunk(2, dim=1)[1]
    block_seq_len = q.shape[1] // 2

    comm = A2AComm(process_group)
    current_rank= comm.rank
    stream_comm=torch.cuda.Stream()

    out_mapping_current_rank = out_mapping[current_rank]
    in_mapping_current_rank = in_mapping[current_rank]
    source_rank_per_iter_current_rank = source_rank_per_iter[current_rank]

    out = None
    lse=None

    para_size=world_size-1

    this_kv=[torch.stack((k[i],v[i]), dim=0) for i in range(para_size)]
    k, v= None, None
    next_kv= [torch.empty_like(this_kv[0]) for _ in range(para_size)]

    #### profile part start ####
    total_start_event = torch.cuda.Event(enable_timing=True)
    total_end_event = torch.cuda.Event(enable_timing=True)
    
    step_events = []
    for step in range(world_size):
        step_events.append({
            'comm_start': torch.cuda.Event(enable_timing=True),
            'comm_end': torch.cuda.Event(enable_timing=True),
            'comp_start': torch.cuda.Event(enable_timing=True),
            'comp_end': torch.cuda.Event(enable_timing=True),
        })
    #### profile part end ####

    def forward(q, k, v,causal):
        outs = _flash_attn_forward(
            q=q,
            k=k,
            v=v,
            dropout_p=0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False
        )
        if len(outs)==8:
            block_out, _, _, _, _, block_softmax_lse, _, _, = outs
        elif len(outs)==4:
            block_out, block_softmax_lse, _, _, = outs

        return block_out, block_softmax_lse
    
    dim_seqlen=1 if layout=="BSND" else 2
    
    # pre-allocate buffer
    Mk_concat_buffer = None
    Mv_concat_buffer = None
    Pk_concat_buffer = None
    Pv_concat_buffer = None


    #### profile part start ####
    total_start_event.record()
    #### profile part end ####

    
    for step in range(comm.world_size):

        k= [this_kv[i][0] for i in range(para_size)]
        v= [this_kv[i][1] for i in range(para_size)]

        with torch.cuda.stream(stream_comm):

            #### profile part start ####
            step_events[step]['comm_start'].record()
            #### profile part end ####
            if step + 1 != comm.world_size:
                # fetch KV chunks from other ranks

                kv_send=[
                     this_kv[out_mapping_current_rank[i]] \
                        if out_mapping_current_rank[i]!=-1 else torch.empty(0, dtype=q.dtype, device=q.device) \
                            for i in range(world_size)]
                kv_recv=[
                        next_kv[in_mapping_current_rank[i]] \
                        if in_mapping_current_rank[i]!=-1 else torch.empty(0, dtype=q.dtype, device=q.device) \
                            for i in range(world_size)]
                comm.all_to_all(kv_send,kv_recv)
                comm.wait()
            
            #### profile part start ####
            step_events[step]['comm_end'].record()
            #### profile part end ####

        if step == 0 :
            #### profile part start ####
            step_events[step]['comp_start'].record()
            #### profile part end ####

            # causal mask
            k_split=[None]*(2*para_size)
            v_split=[None]*(2*para_size)
            for idx00,(k_slice,v_slice) in enumerate(zip(k,v)):
                
                k_chunks0=k_slice.chunk(2,dim=dim_seqlen)
                k_split[idx00]=k_chunks0[0]
                k_split[idx00+para_size]=k_chunks0[1]

                v_chunks0=v_slice.chunk(2,dim=dim_seqlen)
                v_split[idx00]=v_chunks0[0]
                v_split[idx00+para_size]=v_chunks0[1]
            k_concat = torch.cat(k_split, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
            v_concat = torch.cat(v_split, dim=dim_seqlen)  # (batch_size, seqlen, nheads_k, headdim)
            block_out, block_lse = forward(q, k_concat, v_concat,causal=True)
            out, lse = update_out_and_lse(out, lse,block_out, block_lse)

            #### profile part start ####
            step_events[step]['comp_end'].record()
            #### profile part end ####
        else:

            #### profile part start ####
            step_events[step]['comp_start'].record()
            #### profile part end ####
            
            # assert step<len(source_rank_per_iter_current_rank), f"step {step} is list index out of range {len(source_rank_per_iter_current_rank)}"
            source_rank= source_rank_per_iter_current_rank[step]
            Mk_list=[]
            Mv_list=[]
            Pk_list=[]
            Pv_list=[]
            # query can be divided into two parts A, B
            # received KV chunks can be divided into 2(n-1) parts, each one of the n-1 parts can be divided into two parts M, N
            for idx01 in range(para_size):
                # M are passed from source ranks that have smaller rank id than current rank 
                if source_rank[idx01]:
                    Mk_list.append(k[idx01].chunk(2,dim=1)[0])
                    Mv_list.append(v[idx01].chunk(2,dim=1)[0])
                else:
                    Pk_list.append(k[idx01])
                    Pv_list.append(v[idx01])
            
            # use pre-allocated buffers
            if len(Mk_list)>0:
                # reuse or re-allocate buffers
                if Mk_concat_buffer is None or Mk_concat_buffer.shape[dim_seqlen] != sum(m.shape[dim_seqlen] for m in Mk_list):
                    Mk_concat = torch.cat(Mk_list, dim=dim_seqlen)
                    Mv_concat = torch.cat(Mv_list, dim=dim_seqlen)
                    # renew buffers
                    Mk_concat_buffer = Mk_concat
                    Mv_concat_buffer = Mv_concat
                else:
                    # reuse buffers
                    start_idx = 0
                    for i, m in enumerate(Mk_list):
                        end_idx = start_idx + m.shape[dim_seqlen]
                        Mk_concat_buffer.narrow(dim_seqlen, start_idx, m.shape[dim_seqlen]).copy_(m)
                        Mv_concat_buffer.narrow(dim_seqlen, start_idx, Mv_list[i].shape[dim_seqlen]).copy_(Mv_list[i])
                        start_idx = end_idx
                    
                block_out, block_lse = forward(q, Mk_concat_buffer, Mv_concat_buffer, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            
            if len(Pk_list)>0:
                # reuse or re-allocate buffers
                if Pk_concat_buffer is None or Pk_concat_buffer.shape[dim_seqlen] != sum(p.shape[dim_seqlen] for p in Pk_list):
                    Pk_concat = torch.cat(Pk_list, dim=dim_seqlen)
                    Pv_concat = torch.cat(Pv_list, dim=dim_seqlen)
                    # renew buffers
                    Pk_concat_buffer = Pk_concat
                    Pv_concat_buffer = Pv_concat
                else:
                    # reuse buffers
                    start_idx = 0
                    for i, p in enumerate(Pk_list):
                        end_idx = start_idx + p.shape[dim_seqlen]
                        Pk_concat_buffer.narrow(dim_seqlen, start_idx, p.shape[dim_seqlen]).copy_(p)
                        Pv_concat_buffer.narrow(dim_seqlen, start_idx, Pv_list[i].shape[dim_seqlen]).copy_(Pv_list[i])
                        start_idx = end_idx
                
                block_out1, block_lse1 = forward(q1, Pk_concat_buffer, Pv_concat_buffer, causal=False)
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out1,
                    block_lse1,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )
            #### profile part start ####
            step_events[step]['comp_end'].record()
            #### profile part end ####
            

        if step + 1 != comm.world_size:
            # comm.wait()
            stream_comm.synchronize()
            tmp=this_kv
            this_kv=next_kv
            next_kv=tmp

    #### profile part start ####
    total_end_event.record()
    torch.cuda.synchronize()

    total_time = total_start_event.elapsed_time(total_end_event) 
    comm_total_time = 0.0
    comp_total_time = 0.0
    for step in range(world_size):
        comm_time = step_events[step]['comm_start'].elapsed_time(step_events[step]['comm_end'])
        comm_total_time += comm_time
        
        comp_time = step_events[step]['comp_start'].elapsed_time(step_events[step]['comp_end'])
        comp_total_time += comp_time
    #### profile part end ####

    
    out = out.to(q.dtype)
    return out, lse,{
        'total_time': total_time,
        'comm_total_time': comm_total_time,
        'comp_total_time': comp_total_time
    }
    
def hamilton_attention_forward_causal_flash_attn_debug(
    process_group,
    q: torch.Tensor,                # (batch_size, seqlen, nheads, headdim) BSND /  (batch_size, nheads, seqlen, headdim) BNSD 
    k: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    v: list[torch.Tensor],          # (batch_size, seqlen, nheads_k, headdim) BSND / (batch_size, nheads_k, seqlen, headdim) BNSD
    softmax_scale0,
    dropout_p=0,
    world_size=8,
    looping=None, 
    out_mapping=None,
    in_mapping=None, 
    source_rank_per_iter=None,
    layout="BSND",
    enable_profiling=False  # add profiling switch on/off
):
    pass

if __name__ == "__main__":
    circle0=gen_hamilton_circle(8)
    print("Circle0:: ",circle0)

    in_mapping=calculate_in_mapping_circle(circle0)
    out_mapping=calculate_out_mapping_circle(circle0)


    print("In_mapping:: ",in_mapping)
    print("Out_mapping:: ",out_mapping)

    mask_index=calculate_mask_index_circle(circle0)

    print("Mask_index:: ",mask_index)

    source_rank_per_iter=calculate_source_rank_per_iter(circle0)

    print("Source_rank_per_iter[rank 1]:: ",source_rank_per_iter[1])
    print("Source_rank_per_iter[rank 3]:: ",source_rank_per_iter[3])
    print("Source_rank_per_iter[rank 5]:: ",source_rank_per_iter[5])
    print("Source_rank_per_iter[rank 7]:: ",source_rank_per_iter[7])


    # test init qkv
    seqlen0=2*8*7

    q=torch.arange(seqlen0).unsqueeze(0)
    k=torch.arange(seqlen0).unsqueeze(0)
    v=torch.arange(seqlen0).unsqueeze(0)

    out=torch.arange(seqlen0).unsqueeze(0)
    lse=torch.arange(seqlen0).unsqueeze(0).unsqueeze(0)

    print("q:: ",q)
    print("k:: ",k)
    print("v:: ",v)

    print("out:: ",out)
    print("lse:: ",lse)



    for rank in range(8):
        local_q, local_k, local_v=get_circle_init_qkv_for_rank_causal(circle0,q,k,v,rank)
        print(f"rank {rank}:: local_q",local_q)
        print(f"rank {rank}:: local_k",local_k)
        print(f"rank {rank}:: local_v",local_v)
        print()
    
        local_out,local_lse = get_circle_local_out_lse_for_rank_causal(circle0, out,lse, rank)

        print(f"rank {rank}:: local_out",local_out)
        print(f"rank {rank}:: local_lse",local_lse)
        print()
        print()

    # test init qkv
    seqlen0=2*2*8*7

    q=torch.arange(seqlen0).unsqueeze(0)
    k=torch.arange(seqlen0).unsqueeze(0)
    v=torch.arange(seqlen0).unsqueeze(0)

    out=torch.arange(seqlen0).unsqueeze(0)
    lse=torch.arange(seqlen0).unsqueeze(0).unsqueeze(0)

    print("q:: ",q)
    print("k:: ",k)
    print("v:: ",v)

    print("out:: ",out)
    print("lse:: ",lse)

    hamilton_circle_two_nodes=gen_hamilton_circle_two_nodes(16)
    print("hamilton_circle_two_nodes:: ", hamilton_circle_two_nodes)
    print()

    path0=gen_hamilton_path_local(8)
    print("path0:: ",path0)
    print()

    intra_node_in_mapping=calculate_intra_node_in_mapping_two_nodes(path0)
    print("intra_node_in_mapping:: ",intra_node_in_mapping)
    print()

    intra_node_out_mapping=calculate_intra_node_out_mapping_two_nodes(path0)
    print("intra_node_out_mapping:: ",intra_node_out_mapping)
    print()
 

