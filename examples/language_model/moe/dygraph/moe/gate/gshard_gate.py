import math
import paddle
import paddle.nn.functional as F
import numpy as np
from  .naive_gate import NaiveGate
from moe.utils import limit_by_capacity,limit_by_capacity_dp

class GShardGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size, 
            topk=2, capacity=(1.2, 2.4), random_routing=False, group=None, topo_scale=None):
        #assert topk == 2, "topk should be 2 in gshard"
        super().__init__(d_model, num_expert, world_size,topk)
        self.capacity = capacity
        self.random_routing = random_routing
        self.group = group
        if topo_scale is not None:
            self.topo_scale = topo_scale
            topo_scale = paddle.multiply(topo_scale.reshape([-1,1]), paddle.ones([world_size, num_expert], dtype=topo_scale.dtype) ).reshape([-1])
            cap_scale = paddle.reciprocal(topo_scale)
            self.cap_scale = paddle.divide(cap_scale,cap_scale.sum())
            self.topo_scale = paddle.reshape(F.softmax(self.topo_scale) * world_size * world_size, [-1, 1])
            self.topo_scale = paddle.multiply(paddle.cast(self.topo_scale, dtype=paddle.float32), paddle.ones([world_size, num_expert]) )
            self.topo_scale = self.topo_scale.reshape_([num_expert*world_size])
        else:
            self.topo_scale = topo_scale

    def forward(self, x):
        topk_val, topk_idx, gate_score = super().forward(x, return_all_scores=True)
        s = gate_score.shape[0]
        top1_idx = topk_idx.flatten()
        top1_idx = paddle.reshape(top1_idx, shape=[len(top1_idx), 1])
        c_e = paddle.scatter_nd_add(
            x=paddle.zeros(shape=[self.tot_expert]),
            index=top1_idx,
            updates=paddle.ones_like(top1_idx, dtype=paddle.float32).reshape(shape=[len(top1_idx)]),
        ) / s
        if self.topo_scale is not None:
             #print(self.topo_scale)
            c_e = c_e * self.topo_scale
        #print(self.topo_scale)
        m_e = paddle.mean(F.softmax(gate_score, axis=1), axis=0)
        loss = paddle.mean(c_e * m_e) * (self.num_expert ** 2)
        self.set_loss(loss)

        cap_rate = self.capacity[0 if self.training else 1]
        #capacity = math.ceil(cap_rate * x.shape[0]) // (self.world_size * self.num_expert)
        #capacity = paddle.ones(shape=[self.num_expert*self.world_size], dtype=paddle.int64) * capacity
        #capacity = paddle.ceil(cap_rate * self.cap_scale * inp.shape[0])
        capacity = math.ceil(cap_rate * x.shape[0])
        _new_lec, _new_gec, topk_idx = limit_by_capacity(
            topk_idx, self.num_expert, self.world_size, capacity, group=self.group
        )

        if self.random_routing:
            assert False
            rand_routing_prob = paddle.rand(shape=[gate_score.shape[0]])
            mask = (2 * topk_val[:, 1] < rand_routing_prob)
            topk_idx[paddle.nonzero(mask), 1] = -1
        #print(_new_lec)
        #return topk_val, topk_idx, _new_lec, _new_gec
        return topk_val, topk_idx

if __name__ == "__main__":
    paddle.seed(2021)
    import paddle.distributed as dist
    dist.init_parallel_env()
    d_model = 2
    num_expert = 2
    world_size = 2
    gate = GShardGate(d_model=d_model, num_expert=num_expert, world_size=world_size, capacity=(1.2, 1.3))
    inp = paddle.to_tensor([
        [0, 1],
        [2, 3],
        [3, 1],
        [2, 3]
    ], dtype="float32")
    out = gate(inp)
    # print(out)
