import collections
import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F 
from paddle.distributed.utils import expert_count, assign_pos, global_scatter, global_gather
from paddle.distributed import alltoall, all_gather

from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.distributed import fleet
from paddle.autograd import PyLayer
from .gate import NaiveGate, GShardGate, SwitchGate, HirGate
from .utils import count_by_gate

def _local_scatter(inp, pos):
    if pos.shape!=[0]:
        inp_buf = paddle.index_select(inp, pos, 0)
    else:
        inp_buf = paddle.empty([0, inp.shape[1]])
    return inp_buf

def _local_gather(inp, pos, out_batch_size, maybe_overlap=True):
    if pos.shape!=[0]:
        pos = pos.reshape_([-1, 1])
        inp_buf = paddle.scatter_nd(pos, inp, [out_batch_size, inp.shape[-1]])
    else:
        inp_buf = paddle.zeros([out_batch_size, inp.shape[-1]])
    return inp_buf

def _all_gather(tensor, group=None, use_calc_stream=True):
    """
    The main difference with paddle.distributed.all_gather: 
    no need to pass in tensor_list, the returned tensor is spliced
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    nranks = paddle.distributed.collective._get_global_group(
    ).nranks if group is None else group.nranks
    return paddle._C_ops.c_allgather(tensor, 'use_calc_stream', use_calc_stream,
                              'ring_id', ring_id, 'nranks', nranks)

class MOEScatter(PyLayer):
    r"""
    Scatter input samples from [batch x sequences] to contiguous alone experts.
    If `world_size` is greater than 1, the samples will first be locally
    scattered, and then exchanged across workers.
    """
    @staticmethod
    def forward(ctx,
                inp,
                pos,
                local_expert_count,
                global_expert_count,
                fwd_batch_size,
                world_size,
                group=None):
        local_input_buf = _local_scatter(inp, pos)
        if world_size > 1:
            global_input_buf = global_scatter(
                local_input_buf,
                local_expert_count,
                global_expert_count,
                group=group
            )
        else:
            global_input_buf = local_input_buf

        ctx.moe_args = inp.shape[0], world_size, group

        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return global_input_buf

    @staticmethod
    def backward(ctx, grad):
        (pos, local_expert_count, global_expert_count) = ctx.saved_tensor()
        (inp_batch_size, world_size, group) = ctx.moe_args

        if world_size > 1:
            local_grad_in = global_gather(
                grad,
                local_expert_count,
                global_expert_count,
                group=group
            )
        else:
            local_grad_in = grad
        grad_in = _local_gather(local_grad_in, pos, inp_batch_size)
        return grad_in, None, None, None


class MOEGather(PyLayer):
    r"""
    Gather output samples from contiguous alone experts back to [batch x
    sequences]. Works symmetrically with MOEScatter.
    """

    @staticmethod
    def forward(ctx,
                global_output_buf,
                pos,
                local_expert_count,
                global_expert_count,
                local_batch_size,
                world_size,
                group=None
               ):
        if world_size > 1:
            local_output_buf = global_gather(
                global_output_buf,
                local_expert_count,
                global_expert_count,
                group=group
            )
        else:
            local_output_buf = global_output_buf
        output = _local_gather(local_output_buf,
                               pos,
                               local_batch_size,
                               maybe_overlap=False)

        ctx.moe_args = (global_output_buf.shape[0], world_size, group)
        variables = (pos, local_expert_count, global_expert_count)
        ctx.save_for_backward(*variables)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        pos, local_expert_count, global_expert_count = ctx.saved_tensor()
        fwd_batch_size, world_size, group = ctx.moe_args
        grad_out_buf = _local_scatter(grad_out, pos)
        if world_size > 1:
            global_grad_out_buf = global_scatter(
                grad_out_buf,
                local_expert_count,
                global_expert_count,
                group=group
            )
        else:
            global_grad_out_buf = grad_out_buf
        return global_grad_out_buf, None, None, None

class AllGather(PyLayer):
    r"""
    A wrapper for the All-Gather function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        tensor_list = []
        paddle.distributed.all_gather(tensor_list, inp, group=group)
        output = paddle.concat(tensor_list, axis=0)
        ctx.args = rank, inp.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_out):
        rank, dim0 = ctx.args
        return paddle.slice(grad_out, axes=[0], starts=[rank*dim0], ends=[(rank+1)*dim0])

class Slice(PyLayer):
    r"""
    A wrapper for the Slice function to support auto-differentiation.
    """

    @staticmethod
    def forward(ctx, inp, rank, world_size, group):
        B = inp.shape[0]
        local_batch_size = B // world_size
        batch_start = local_batch_size * rank
        batch_end = min(batch_start + local_batch_size, B)
        inp = paddle.slice(inp, axes=[0], starts=[batch_start], ends=[batch_end])
        ctx.args = world_size, group
        return inp

    @staticmethod
    def backward(ctx, grad_out):
        world_size, group = ctx.args
        # tensor_list = []
        # paddle.distributed.all_gather(tensor_list, grad_out, group=group)
        # grad_out = paddle.concat(tensor_list, axis=0)
        return _all_gather(grad_out, group=group)
        # return grad_out


def prepare_forward(gate, num_expert, world_size, hcg):
    moe_group = hcg.get_expert_parallel_group()
    #print(gate.dtype)
    pos, local_expert_count, global_expert_count = count_by_gate(gate, 
            num_expert, world_size, group=moe_group)
    #print(global_expert_count)
    with paddle.no_grad():
        fwd_expert_count = global_expert_count.reshape([world_size,
                num_expert]).sum(axis=0)
        print(local_expert_count,global_expert_count)
        fwd_batch_size = int(fwd_expert_count.sum().item())
    return (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    )


class FMoELinear(nn.Layer):
    def __init__(self,
                num_expert,
                in_feat,
                out_feat,
                bias=True,
                rank=0):
        super(FMoELinear, self).__init__()

        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self._dtype = self._helper.get_default_dtype()

        self.weight = self.create_parameter(
                                shape=[num_expert, in_feat, out_feat],
                                attr=nn.initializer.KaimingUniform(),
                                dtype=self._dtype,
                                is_bias=False)
                 
        assert bias is True, "should have bias for moe"
        self.bias = self.create_parameter(
                                shape=[num_expert, out_feat],
                                attr=paddle.nn.initializer.Constant(value=0.0),
                                dtype=self._dtype,
                                is_bias=True)

        self.weight.name = "expert_" + self.weight.name
        self.bias.name = "expert_" + self.bias.name

    def forward(self, inp, fwd_expert_count):
        out = paddle.distributed.utils.parallel_linear(inp,
                                                       self.weight,
                                                       self.bias,
                                                       fwd_expert_count)
        return out

class _Expert(nn.Layer):
    def __init__(self, num_expert, d_model, d_hidden, rank=0):
        super().__init__()
        self.htoh4 = nn.Linear(d_model, d_hidden, name="expert_")
        self.h4toh = nn.Linear(d_hidden, d_model, name="expert_")
        self.htoh4.weight.name = "expert_" + self.htoh4.weight.name
        self.h4toh.weight.name = "expert_" + self.h4toh.weight.name
        self.htoh4.bias.name = "expert_" + self.htoh4.bias.name
        self.h4toh.bias.name = "expert_" + self.h4toh.bias.name

    def forward(self, inp, fwd_expert_count):
        print(inp.shape)
        if inp.shape[0] is not 0:
            x = self.htoh4(inp)
            x = F.gelu(x, approximate=True)
            x = self.h4toh(x)
            return x
        else:
            return inp


class MoeLayer(nn.Layer):
    
    def __init__(self, 
                 total_expert, 
                 local_expert, 
                 d_model,
                 hidden_dim,
                 top_k,
                 weight_attr,
                 bias_attr,
                 hcg=None,
                 timers=None,
                 gate=None,
                 topo_scale=None):
        super(MoeLayer, self).__init__()

        print("total_expert", total_expert, "local_expert", local_expert)
        self.total_expert = total_expert
        self.local_expert = local_expert
        self.hcg = hcg
        assert self.hcg is not None, "self.hcg for MoE layer should not be None."
        self.topo_scale = topo_scale
        # only support mp/dp
        self.group = self.hcg.get_expert_parallel_group()
        self.world_size = self.hcg.get_expert_parallel_world_size()
        self.num_expert = local_expert

        self.hidden_dim = hidden_dim
        self.d_model = d_model

        self.experts = _Expert(
            local_expert, d_model, hidden_dim // 2, rank=0)

        self.top_k = top_k
        if gate == "naive" or gate is None:
            self.gate = NaiveGate(d_model, num_expert=self.num_expert, world_size=self.world_size, topk=self.top_k)
        elif gate == "gshard":
            self.gate = GShardGate(d_model, num_expert=self.num_expert, world_size=self.world_size, topk=self.top_k, group=self.group, topo_scale=self.topo_scale)
        elif gate == "switch":
            self.gate = SwitchGate(d_model, num_expert=self.num_expert, world_size=self.world_size, topk=self.top_k, group=self.group)
        elif gate == "hir":
            self.gate = HirGate(d_model, n_expert=self.num_expert, world_size=self.world_size, node_rank=self.hcg.get_data_parallel_rank()//8)
        else:
            assert False, "We only support naive gate, gshard gate and switch gate, but you choose {} gate.".format(str(gate))
        self.hcg = hcg
        self.mp_rank = self.hcg.get_model_parallel_rank()
        self.mp_size = self.hcg.get_model_parallel_world_size()
        self.mp_group = self.hcg.get_model_parallel_group()
        self.timers = timers

    def forward(self, inp):
        # inp shape: b * s * m
        assert len(inp.shape) == 3
        origin_shape = inp.shape
        inp = inp.reshape_([-1, origin_shape[2]])

        mp_rank = self.hcg.get_model_parallel_rank()
        mp_size = self.hcg.get_model_parallel_world_size()
        mp_group = self.hcg.get_model_parallel_group()
        if mp_size > 1: 
            inp = Slice.apply(inp, mp_rank, mp_size, mp_group)
        self.timers('Gate Computation').start()
        
        # baseline
        value, gate = self.gate(inp)
        self.timers('Gate Computation').stop()
        self.timers('Prepare Forward').start()
        
        (
            pos,
            local_expert_count,
            global_expert_count,
            fwd_expert_count,
            fwd_batch_size,
        ) = prepare_forward(gate, self.num_expert, self.world_size, self.hcg)
        # not baseline
        '''
        value, gate, local_expert_count, global_expert_count= self.gate(inp)
        self.timers('Gate Computation').stop()
        self.timers('Prepare Forward').start()
        lec_cum = paddle.cumsum(local_expert_count, axis=0)
        pos = assign_pos(gate, lec_cum)
        fwd_expert_count = global_expert_count.reshape_([self.world_size, self.num_expert]).sum(axis=0)
        fwd_batch_size = int(fwd_expert_count.sum().item())
        '''
        self.timers('Prepare Forward').stop()
        #print(fwd_batch_size)
        #print('{},{},{}'.format(local_expert_count, global_expert_count, fwd_batch_size))
        topk = 1
        if len(gate.shape) == 2:
            topk = gate.shape[1]
        
        if pos.shape != [0] :
            temp_pos = pos // topk
        else:
            temp_pos = pos
        assert topk == self.top_k
        self.timers('MOEScatter').start()
        x = MOEScatter.apply(
            inp, temp_pos,
            local_expert_count, global_expert_count, fwd_batch_size, self.world_size, self.group
        )
        self.timers('MOEScatter').stop()
        self.timers('Expert Computation').start()
        x = self.experts(x, fwd_expert_count.numpy())
        self.timers('Expert Computation').stop()
        out_batch_size = inp.shape[0]
        if len(gate.shape) == 2:
            out_batch_size *= gate.shape[1]
        self.timers('MOEGather').start()
        x = MOEGather.apply(
            x, pos,
            local_expert_count, global_expert_count,
            out_batch_size, self.world_size, self.group
        )
        self.timers('MOEGather').stop()
        x = x.reshape([-1, self.top_k, self.d_model])
        self.timers('Score BMM').start()
        value = value.reshape([x.shape[0], 1, self.top_k])
        x = paddle.bmm(value, x).reshape([-1, self.d_model])
        self.timers('Score BMM').stop()
        self.timers('AllGather').start()
        if mp_size > 1:
            x = AllGather.apply(x, mp_rank, mp_size, mp_group)

        x = paddle.reshape_(x, origin_shape)
        self.timers('AllGather').stop()
        return x
 
