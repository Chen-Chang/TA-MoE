import paddle
from paddle.distributed.utils import expert_count, assign_pos

def _alltoall(in_tensor_list, group=None, use_calc_stream=True):
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    nranks = len(in_tensor_list)
    return paddle._C_ops.alltoall(in_tensor_list, 'use_calc_stream', use_calc_stream,
                              'ring_id', ring_id)

def count_by_gate(gate, num_expert, world_size, require_pos=True, group=None):
    total_expert_count = num_expert * world_size
    with paddle.no_grad():
        local_expert_count = expert_count(gate, total_expert_count)

        if world_size > 1:
            global_expert_count = _alltoall(local_expert_count, group=group)
        else:
            global_expert_count = local_expert_count
        if not require_pos:
            pos = None
        else:
            lec_cum = paddle.cumsum(local_expert_count, axis=0)
            pos = assign_pos(gate, lec_cum)
    return pos, local_expert_count, global_expert_count


def limit_by_capacity(topk_idx, num_expert, world_size, capacity, group=None):
    with paddle.no_grad():
        capacity = paddle.ones(shape=[num_expert], dtype=paddle.int64) * capacity
        pos, lec, gec = count_by_gate(topk_idx, num_expert, world_size,
                        require_pos=False, group=group)
        #print(gec)
        new_gec = paddle.distributed.utils.limit_by_capacity(gec, capacity, world_size)
        if world_size > 1:
            new_lec = []
            paddle.distributed.alltoall(
                paddle.split(new_gec, world_size, axis=0),
                new_lec, group=group
            )
            new_lec = paddle.concat(new_lec, axis=0)
        else:
            new_lec = new_gec
        new_lec = paddle.to_tensor(new_lec, dtype="int64")
        topk_idx = paddle.distributed.utils.prune_gate_by_capacity(topk_idx, new_lec, num_expert, world_size)

    return new_lec, new_gec, topk_idx


def limit_by_capacity_dp(topk_idx, num_expert, world_size, capacity, group=None):
    
    with paddle.no_grad():
        local_expert_count = expert_count(topk_idx, num_expert*world_size)
        gshard_count = local_expert_count

        local_expert_count = paddle.minimum(paddle.cast(capacity, dtype=local_expert_count.dtype).reshape_([-1]),
                            local_expert_count)
        topk_idx = paddle.distributed.utils.prune_gate_by_capacity(topk_idx, local_expert_count, num_expert, world_size)

        if world_size > 1:
            global_expert_count = _alltoall(local_expert_count, group=group)
        else:
            global_expert_count = local_expert_count

    return local_expert_count, global_expert_count, topk_idx

    

    
    
