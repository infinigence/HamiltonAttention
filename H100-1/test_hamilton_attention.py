"""
torchrun --nproc_per_node=8 <path-to-this-script>
"""

"""
git push -u origin1 "main"
"""

import torch
import os
import errno
import random
import json
import torch.nn.functional as F

import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function

from flash_attn.flash_attn_interface import _flash_attn_forward


from hamilton_attention import hamilton_attention_forward_full_flash_attn,\
    gen_hamilton_circle,\
    calculate_in_mapping_circle,\
    calculate_out_mapping_circle,\
    get_circle_init_qkv_for_rank,\
    get_circle_local_out_lse_for_rank,\
    calculate_source_rank_per_iter,\
    get_circle_init_qkv_for_rank_causal,\
    get_circle_local_out_lse_for_rank_causal,\
    hamilton_attention_forward_causal_flash_attn,\
    hamilton_attention_forward_causal_flash_attn_debug,\
    hamilton_attention_forward_full_flash_attn_profile,\
    hamilton_attention_forward_causal_flash_attn_profile,\
    extract_local_for_zigzag_ring,\
    extract_local_for_ring,\
    ring_attention_forward_flash_attn,\
    ring_attention_forward_flash_attn_profile,\
    zigzag_ring_attention_forward_flash_attn,\
    zigzag_ring_attention_forward_flash_attn_profile,\
    get_circle_init_qkv_for_rank_dummy,\
    extract_local_for_ring_dummy,\
    hamilton_attention_forward_full_flash_attn_two_nodes,\
    hamilton_attention_forward_full_flash_attn_two_nodes_profile,\
    gen_hamilton_path_local,\
    gen_hamilton_circle_two_nodes,\
    calculate_intra_node_in_mapping_two_nodes,\
    calculate_intra_node_out_mapping_two_nodes,\
    calculate_inter_node_mapping
    

def safe_create_dir(path0):
    try:
        os.makedirs(path0, mode=0o777, exist_ok=False)
    except OSError as e:
        if e.errno != errno.EEXIST:  
            raise
    return path0

def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_dict(msg, dict0):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            dict_str = ", ".join([f"{k}: {v}" for k, v in dict0.items()])
            print(f"[{rank}] {dict_str}", flush=True)
        dist.barrier()

def log_variable(msg,a):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f" {a}",
                flush=True,
            )
        dist.barrier()

def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: "
                f"max {a.abs().max().item():.3g}, "
                f"mean {a.abs().mean().item():.3g}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] "
                f"max {a.abs().max().item():.3g}, "
                f"mean {a.abs().mean().item():.3g}",
                flush=True,
            )
        dist.barrier()



def init_all():
    local_rank = int(os.environ['LOCAL_RANK'])  
    global_rank = int(os.environ['RANK'])

    # Set the current GPU device
    torch.random.manual_seed(2023)
    torch.cuda.set_device(local_rank)
    device_curr=torch.device(f'cuda:{local_rank}')

    dist.init_process_group(backend="nccl",device_id=device_curr)   
    return local_rank,global_rank   

def prof_perform(
    bs,
    seqlen,
    nheads,
    d,
    causal,
    two_nodes,
    test_nums,
    warm_ups,
    local_rank,
    log_all=False
    ):


  
    rank = dist.get_rank()                  
    set_seed(rank)                         
    world_size = dist.get_world_size()      
    

    dtype = torch.bfloat16                  
    device = torch.device(f"cuda:{local_rank}")   

    batch_size = bs                         
    seqlen = seqlen
    nheads = nheads                              
    d = 64                                  
    dropout_p = 0                           
    
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    storage_prefix=f"{current_dir}/traces"

    
    causal = causal        
    profile_version=True
    accuracy_test=False
    torch_profile_on=False

    two_nodes=(two_nodes and world_size==16 and causal==False)                 


    if causal:
        name_hamilton=f"K{world_size}-zigzag-hamilton"
        name_ring = f"R{world_size}-zigzag-ring"
    else:
        name_hamilton=f"K{world_size}-hamilton" if not two_nodes else "K8-16-K8-hamilton"
        name_ring = f"R{world_size}-ring"

    res={
        "hamilton":
        {
            "name":name_hamilton,
            "batch_size":bs,
            "seqlen":seqlen,
            "nheads":nheads,
            "d":d,
            "causal":causal,
            "two_nodes":two_nodes,
            "test_nums":test_nums,
            "warm_ups":warm_ups
        },
        "ring":
        {
            "name":name_ring,
            "batch_size":bs,
            "seqlen":seqlen,
            "nheads":nheads,
            "d":d,
            "causal":causal,
            "test_nums":test_nums,
            "warm_ups":warm_ups
        }
    }

    debug = False
    debug_rank=1

    test_nums=test_nums
    warm_ups=warm_ups
    
    assert seqlen % world_size == 0

    if not debug:
        if accuracy_test:
            q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
            k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
            v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
            dist.broadcast(q, src=0)    
            dist.broadcast(k, src=0)
            dist.broadcast(v, src=0)
    else:

        seqlen=4*world_size*(world_size-1)
        q= torch.arange(seqlen,device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        k= torch.arange(seqlen,device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        v= torch.arange(seqlen,device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        if rank == debug_rank:
            print(f"rank:: {rank}, q:: ",q)
            print(f"rank:: {rank}, k:: ",k)
            print(f"rank:: {rank}, v:: ",v)

        dist.broadcast(q, src=0)    
        dist.broadcast(k, src=0)
        dist.broadcast(v, src=0)

    if not two_nodes:
        looping= gen_hamilton_circle(world_size)
        out_mapping = calculate_out_mapping_circle(looping)
        in_mapping= calculate_in_mapping_circle(looping)
        if causal:
            source_rank_per_iter=calculate_source_rank_per_iter(looping)
    else:
        looping=gen_hamilton_circle_two_nodes(16)
        local_path=gen_hamilton_path_local(8)
        out_mapping=calculate_intra_node_out_mapping_two_nodes(local_path)
        in_mapping=calculate_intra_node_in_mapping_two_nodes(local_path)
        inter_in_mapping, inter_out_mapping=calculate_inter_node_mapping(looping)


    if not causal:
        if accuracy_test:
            local_q, local_k, local_v=get_circle_init_qkv_for_rank(looping,q,k,v,rank)
        else:
            local_q, local_k, local_v=get_circle_init_qkv_for_rank_dummy(
                looping,
                batch_size,
                seqlen,
                nheads,
                d,
                device,
                dtype
            )
    else:
        if accuracy_test:
            local_q, local_k, local_v=get_circle_init_qkv_for_rank_causal(looping,q,k,v,rank)
        else:
            local_q, local_k, local_v=get_circle_init_qkv_for_rank_dummy(
                looping,
                batch_size,
                seqlen,
                nheads,
                d,
                device,
                dtype
            )


    if debug and rank == debug_rank:
        print(f"local_q shape: {local_q.shape}")
        print(f"local_k shape: {local_k[0].shape}")
        print(f"local_v shape: {local_v[0].shape}")
        print(f"rank:: {rank}, local_q:: ",local_q)
        print(f"rank:: {rank}, local_k:: ",local_k)
        print(f"rank:: {rank}, local_v:: ",local_v)

    dist.barrier()  

    

    if not debug:
        if accuracy_test:

            softmax_scale=q.shape[-1] ** (-0.5)
            outs=_flash_attn_forward(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=softmax_scale ,
                causal=causal,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False
            )

            if len(outs)==8:
                out,_,_,_,_,lse,_,_ =outs
            elif len(outs)==4:
                out,lse,_,_ =outs



            if not causal:
                local_out,local_lse = get_circle_local_out_lse_for_rank(looping, out,lse, rank)
                local_lse=local_lse.transpose(-2,-1)
            else:
                local_out,local_lse = get_circle_local_out_lse_for_rank_causal(looping, out,lse, rank)
                local_lse=local_lse.transpose(-2,-1)




    dist_out=[]
    dist_lse=[]
    profile_dict=[]
    
    if not debug:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            device = "cuda"
            activities += [ProfilerActivity.CUDA]
        elif torch.xpu.is_available():
            device = "xpu"
            activities += [ProfilerActivity.XPU]
        else:
            print(
            "Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices"
            )   
            import sys
            sys.exit(0)
        sched0=torch.profiler.schedule(
            wait=0,    
            warmup=warm_ups,    
            active=test_nums,   
            repeat=1     
        )
        if torch_profile_on:
            prof0=profile(activities=activities,schedule=sched0) 
            prof0.start()
        for iter0 in range(test_nums+warm_ups):
            with record_function(f"test hamilton::{iter0}"):
                if not causal:
                    if not profile_version:
                        if accuracy_test:
                            if not two_nodes:
                                dist_out_tmp, dist_lse_tmp=hamilton_attention_forward_full_flash_attn(
                                    None,
                                    local_q,
                                    local_k,
                                    local_v,
                                    None,
                                    world_size=world_size,
                                    looping=looping,
                                    out_mapping=out_mapping,
                                    in_mapping=in_mapping,
                                )
                                dist_out.append(dist_out_tmp)
                                dist_lse.append(dist_lse_tmp)
                            else:
                                dist_out_tmp, dist_lse_tmp=hamilton_attention_forward_full_flash_attn_two_nodes(
                                    None,
                                    local_q,
                                    local_k,
                                    local_v,
                                    None,
                                    inter_out_mapping=inter_out_mapping,
                                    inter_in_mapping=inter_in_mapping,
                                    out_mapping=out_mapping,
                                    in_mapping=in_mapping,
                                )
                                dist_out.append(dist_out_tmp)
                                dist_lse.append(dist_lse_tmp)
                        else:
                            if not two_nodes:
                                hamilton_attention_forward_full_flash_attn(
                                    None,
                                    local_q,
                                    local_k,
                                    local_v,
                                    None,
                                    world_size=world_size,
                                    looping=looping,
                                    out_mapping=out_mapping,
                                    in_mapping=in_mapping,
                                )
                            else:
                                hamilton_attention_forward_full_flash_attn_two_nodes(
                                    None,
                                    local_q,
                                    local_k,
                                    local_v,
                                    None,
                                    inter_out_mapping=inter_out_mapping,
                                    inter_in_mapping=inter_in_mapping,
                                    out_mapping=out_mapping,
                                    in_mapping=in_mapping,
                                )
                    else:
                        if accuracy_test:
                            if not two_nodes:
                                dist_out_tmp, dist_lse_tmp, profile_dict_tmp=hamilton_attention_forward_full_flash_attn_profile(
                                    None,
                                    local_q,
                                    local_k,
                                    local_v,
                                    None,
                                    world_size=world_size,
                                    looping=looping,
                                    out_mapping=out_mapping,
                                    in_mapping=in_mapping,
                                )
                                dist_out.append(dist_out_tmp)
                                dist_lse.append(dist_lse_tmp)
                                profile_dict.append(profile_dict_tmp)
                            else:
                                dist_out_tmp, dist_lse_tmp, profile_dict_tmp=hamilton_attention_forward_full_flash_attn_two_nodes_profile(
                                    None,
                                    local_q,
                                    local_k,
                                    local_v,
                                    None,
                                    inter_out_mapping=inter_out_mapping,
                                    inter_in_mapping=inter_in_mapping,
                                    out_mapping=out_mapping,
                                    in_mapping=in_mapping,
                                )
                                dist_out.append(dist_out_tmp)
                                dist_lse.append(dist_lse_tmp)
                                profile_dict.append(profile_dict_tmp)
                        else:
                            if not two_nodes:
                                _, _, profile_dict_tmp=hamilton_attention_forward_full_flash_attn_profile(
                                    None,
                                    local_q,
                                    local_k,
                                    local_v,
                                    None,
                                    world_size=world_size,
                                    looping=looping,
                                    out_mapping=out_mapping,
                                    in_mapping=in_mapping,
                                )
                                profile_dict.append(profile_dict_tmp)
                            else:
                                _, _, profile_dict_tmp=hamilton_attention_forward_full_flash_attn_two_nodes_profile(
                                    None,
                                    local_q,
                                    local_k,
                                    local_v,
                                    None,
                                    inter_out_mapping=inter_out_mapping,
                                    inter_in_mapping=inter_in_mapping,
                                    out_mapping=out_mapping,
                                    in_mapping=in_mapping,
                                )
                                profile_dict.append(profile_dict_tmp)
                else:
                    if not profile_version:
                        if accuracy_test:
                            dist_out_tmp, dist_lse_tmp=hamilton_attention_forward_causal_flash_attn(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                                world_size=world_size,
                                looping=looping,
                                out_mapping=out_mapping,
                                in_mapping=in_mapping,
                                source_rank_per_iter=source_rank_per_iter
                            )
                            dist_out.append(dist_out_tmp)
                            dist_lse.append(dist_lse_tmp)
                        else:
                            hamilton_attention_forward_causal_flash_attn(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                                world_size=world_size,
                                looping=looping,
                                out_mapping=out_mapping,
                                in_mapping=in_mapping,
                                source_rank_per_iter=source_rank_per_iter
                            )
                    else:
                        if accuracy_test:
                            dist_out_tmp, dist_lse_tmp,profile_dict_tmp=hamilton_attention_forward_causal_flash_attn_profile(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                                world_size=world_size,
                                looping=looping,
                                out_mapping=out_mapping,
                                in_mapping=in_mapping,
                                source_rank_per_iter=source_rank_per_iter
                            )
                            dist_out.append(dist_out_tmp)
                            dist_lse.append(dist_lse_tmp)
                            profile_dict.append(profile_dict_tmp)
                        else:
                            _, _, profile_dict_tmp=hamilton_attention_forward_causal_flash_attn_profile(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                                world_size=world_size,
                                looping=looping,
                                out_mapping=out_mapping,
                                in_mapping=in_mapping,
                                source_rank_per_iter=source_rank_per_iter
                            )
                            profile_dict.append(profile_dict_tmp)          

                dist.barrier() 
                if torch_profile_on:
                    prof0.step()
        if torch_profile_on:
            prof0.stop()

        res["hamilton"]["data"]=profile_dict[warm_ups:]

        if log_all:
            for iter0 in range(warm_ups,test_nums+warm_ups,1):
                if accuracy_test:
                    log(f"hamilton out diff {iter0}", local_out - dist_out[iter0])
                    cosine_sim_out=F.cosine_similarity(torch.flatten(local_out),torch.flatten(dist_out[iter0]),dim=0)
                    log_variable(f"hamilton cosine_sim_out {iter0}",cosine_sim_out)
                if profile_version:
                    log_dict(f"hamilton profile stats {iter0}",profile_dict[iter0])
                if rank ==0:
                    print()
        



        
        if torch_profile_on:
            sub_dir0=name_hamilton
            sub_dir1="causal" if causal else "full"
            sub_dir2="profile" if profile_version else "no_profile" 
            full_path_prefix=f"{storage_prefix}/{sub_dir0}/{sub_dir1}/{sub_dir2}"

            safe_create_dir(full_path_prefix)
            prof0.export_chrome_trace(f"{full_path_prefix}/trace_{sub_dir0}_{sub_dir1}_{sub_dir2}_{local_rank}.json")
        

        
        if not causal:
            if accuracy_test:
                local_q=extract_local_for_ring(q,rank,world_size,1)
                local_k=extract_local_for_ring(k,rank,world_size,1)
                local_v=extract_local_for_ring(v,rank,world_size,1)
                local_out=extract_local_for_ring(out,rank,world_size,1)
                local_lse=extract_local_for_ring(lse,rank,world_size,2).transpose(-2,-1)
            else:
                local_q=extract_local_for_ring_dummy(
                    world_size,
                    batch_size,
                    seqlen,
                    nheads,
                    d,
                    device,
                    dtype
                )
                local_k=extract_local_for_ring_dummy(
                    world_size,
                    batch_size,
                    seqlen,
                    nheads,
                    d,
                    device,
                    dtype
                )
                local_v=extract_local_for_ring_dummy(
                    world_size,
                    batch_size,
                    seqlen,
                    nheads,
                    d,
                    device,
                    dtype
                )
        else:
            if accuracy_test:
                local_q=extract_local_for_zigzag_ring(q,rank,world_size,1)
                local_k=extract_local_for_zigzag_ring(k,rank,world_size,1)
                local_v=extract_local_for_zigzag_ring(v,rank,world_size,1)
                local_out=extract_local_for_zigzag_ring(out,rank,world_size,1)
                local_lse=extract_local_for_zigzag_ring(lse,rank,world_size,2).transpose(-2,-1)
            else:
                local_q=extract_local_for_ring_dummy(
                    world_size,
                    batch_size,
                    seqlen,
                    nheads,
                    d,
                    device,
                    dtype
                )
                local_k=extract_local_for_ring_dummy(
                    world_size,
                    batch_size,
                    seqlen,
                    nheads,
                    d,
                    device,
                    dtype
                )
                local_v=extract_local_for_ring_dummy(
                    world_size,
                    batch_size,
                    seqlen,
                    nheads,
                    d,
                    device,
                    dtype
                )

        dist_out=[]
        dist_lse=[]
        profile_dict1=[]

        if torch_profile_on:
            prof1=profile(activities=activities,schedule=sched0)
            prof1.start()
        for iter0 in range(test_nums+warm_ups):
            with record_function(f"test ring::{iter0}"):
                if not causal:
                    if not profile_version:
                        if accuracy_test:
                            dist_out_tmp, dist_lse_tmp=ring_attention_forward_flash_attn(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                            )
                            dist_out.append(dist_out_tmp)
                            dist_lse.append(dist_lse_tmp)
                        else:
                            ring_attention_forward_flash_attn(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                            )
                    else:
                        if accuracy_test:
                            dist_out_tmp, dist_lse_tmp,profile_dict_tmp=ring_attention_forward_flash_attn_profile(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                                world_size=world_size,
                            )
                            dist_out.append(dist_out_tmp)
                            dist_lse.append(dist_lse_tmp)
                            profile_dict1.append(profile_dict_tmp)
                        else:
                            _, _, profile_dict_tmp=ring_attention_forward_flash_attn_profile(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                                world_size=world_size,
                            )
                            profile_dict1.append(profile_dict_tmp)                 
                else:
                    if not profile_version:
                        if accuracy_test:
                            dist_out_tmp, dist_lse_tmp=zigzag_ring_attention_forward_flash_attn(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                            )
                            dist_out.append(dist_out_tmp)
                            dist_lse.append(dist_lse_tmp)
                        else:
                            zigzag_ring_attention_forward_flash_attn(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                            )
                    else:
                        if accuracy_test:
                            dist_out_tmp, dist_lse_tmp,profile_dict_tmp=zigzag_ring_attention_forward_flash_attn_profile(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                                world_size=world_size,
                            )
                            dist_out.append(dist_out_tmp)
                            dist_lse.append(dist_lse_tmp)
                            profile_dict1.append(profile_dict_tmp)
                        else:
                            _, _, profile_dict_tmp=zigzag_ring_attention_forward_flash_attn_profile(
                                None,
                                local_q,
                                local_k,
                                local_v,
                                None,
                                world_size=world_size,
                            )
                            profile_dict1.append(profile_dict_tmp)
                dist.barrier()  
                if torch_profile_on:
                    prof1.step()
        if torch_profile_on:
            prof1.stop()

        if log_all:
            for iter0 in range(warm_ups,test_nums+warm_ups,1):
                if accuracy_test:
                    log(f"ring out diff {iter0}", local_out - dist_out[iter0])
                    cosine_sim_out=F.cosine_similarity(torch.flatten(local_out),torch.flatten(dist_out[iter0]),dim=0)
                    log_variable(f"ring cosine_sim_out {iter0}",cosine_sim_out)
                if profile_version:
                    log_dict(f"ring profile stats {iter0}",profile_dict1[iter0])
                if rank ==0:
                    print()
        
        res["ring"]["data"]=profile_dict1[warm_ups:]


        if torch_profile_on:
            sub_dir0=name_ring
            sub_dir1="causal" if causal else "full"
            sub_dir2="profile" if profile_version else "no_profile" 
            full_path_prefix=f"{storage_prefix}/{sub_dir0}/{sub_dir1}/{sub_dir2}"

            safe_create_dir(full_path_prefix)
            prof1.export_chrome_trace(f"{full_path_prefix}/trace_{sub_dir0}_{sub_dir1}_{sub_dir2}_{local_rank}.json")




    else:
        dist_out, dist_lse= hamilton_attention_forward_causal_flash_attn_debug(
                    None,
                    local_q,
                    local_k,
                    local_v,
                    None,
                    looping=looping,
                    out_mapping=out_mapping,
                    in_mapping=in_mapping,
                    source_rank_per_iter=source_rank_per_iter,
                    debug_rank=debug_rank
                )
    
    if rank==0:
        print(f"finish bs_{bs} seqlen_{seqlen}")
        print()
    return res

def load_existing_results(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_checkpoint(checkpoint_path, completed_tasks):
    with open(checkpoint_path, 'w') as f:
        json.dump(list(completed_tasks), f)

def get_task_signature(seqlen0, bs0, nhead, d):
    return f"{seqlen0}_{bs0}_{nhead}_{d}"

if __name__ == "__main__":
    platform = "NV-1"
    node_num = 1
    causal = False
    two_nodes = False

    test_nums = 30
    warm_ups = 20
    max_len = 10000000

    basic_len = 3360
    seqlens = [128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1]
    bs = [128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1]
    nheads_d = [
        (20, 128),
        (4, 512),
        (12, 192),
    ]

    local_rank, global_rank = init_all()

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    storage_prefix = f"{current_dir}/prof_result"
    
    causal_name = "causal" if causal else "full"
    hello = "" if not two_nodes else "two_nodes/"
    full_path_prefix = f"{storage_prefix}/{platform}/{causal_name}/node_num_{node_num}/{hello}"
    safe_create_dir(full_path_prefix)

    results_file = f'{full_path_prefix}results_{global_rank}.json'
    checkpoint_file = f'{full_path_prefix}checkpoint_{global_rank}.json'

    results = load_existing_results(results_file)
    completed_tasks = load_checkpoint(checkpoint_file)

    print(f"Rank {global_rank}: finish {len(completed_tasks)} tasks, continue")

    all_tasks = []
    for seqlen0 in seqlens:
        for bs0 in bs:
            for nhead, d in nheads_d:
                true_seqlen = seqlen0 * basic_len
                size_all = true_seqlen * bs0
                if size_all > max_len:
                    continue
                
                task_sig = get_task_signature(seqlen0, bs0, nhead, d)
                all_tasks.append((seqlen0, bs0, nhead, d, task_sig))

    pending_tasks = [task for task in all_tasks if task[4] not in completed_tasks]
    
    print(f"Rank {global_rank}: #total-task {len(all_tasks)} #pending-task {len(pending_tasks)}")

    for i, (seqlen0, bs0, nhead, d, task_sig) in enumerate(pending_tasks):
        true_seqlen = seqlen0 * basic_len
        
        print(f"Rank {global_rank}: do task {i+1}/{len(pending_tasks)}: "
              f"seqlen0={seqlen0}, bs0={bs0}, nhead={nhead}, d={d}")
        
        try:
            result = prof_perform(
                bs0, true_seqlen, nhead, d, causal, 
                two_nodes, test_nums, warm_ups, local_rank
            )
            
            result['task_signature'] = task_sig
            results.append(result)
            
            with open(results_file, 'w') as json_file:
                json.dump(results, json_file, indent=4)
            
            completed_tasks.add(task_sig)
            save_checkpoint(checkpoint_file, completed_tasks)
            
            print(f"Rank {global_rank}: complete and saved")
            
        except Exception as e:
            print(f"Rank {global_rank}: fail: {e}")
            completed_tasks.add(task_sig)
            save_checkpoint(checkpoint_file, completed_tasks)
            continue

    print(f"Rank {global_rank}: finish all tasks")
    dist.destroy_process_group()

