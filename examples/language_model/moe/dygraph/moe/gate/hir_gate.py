from .naive_gate import NaiveGate
import sys
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from moe.utils import limit_by_capacity, count_by_gate


nw_per_node = 8

cr_cnt = 0
llims = [3, 2, 5, 2, 8, 2]

def _gen_policy(alpha):
    def _global_policy(all_experts_count, all_global_expert_count, num_expert, world_size, d_model, fused):
        bw_pcie = 88 * 1e9 / 8
        bw_net = 50 * 1e9 / 8
        bw_mm = 11.5e12
        data_size = 4 # TODO data different than float

        if fused:
            all_experts_count = all_experts_count.sum(axis=-1).reshape([world_size, world_size, 1])
            all_global_expert_count = all_global_expert_count.sum(axis=-1).reshape([world_size, world_size, 1])

        fwd_expert_counts = all_global_expert_count.sum(1) # [world_size, num_expert]
        default_counts = fwd_expert_counts.clone()

        indices = fwd_expert_counts.argsort(0, descending=True)

        alphaHsquared = alpha * d_model ** 2 * data_size

        B_w = default_counts.max(0)[0]
        lat_comp = 3 * 4 * B_w * alphaHsquared / bw_mm  + 4 * B_w * d_model * data_size / bw_net

        comm = float('+inf')
        model_size = 2 * alphaHsquared * num_expert / bw_net * 2
        comp_time = 12 * alphaHsquared / bw_mm

        for i, index in enumerate(indices):
            fwd_expert_counts[index] = 0
            fwd_expert_counts += all_global_expert_count[index].reshape([world_size, -1])

            B_k = fwd_expert_counts.max(0)[0]
            lat_comm = fwd_expert_counts.max(0)[0] * comp_time + (i+1) * model_size

            if lat_comm < comm:
                comm = lat_comm
            elif lat_comm > comm:
                break

        res = paddle.zeros([world_size, num_expert], dtype=paddle.bool)

        if lat_comp > comm:
            res[indices[:i]] = True
        return res

    def _no_policy(all_experts_count, all_global_expert_count, num_expert, world_size, d_model, fused):
        if fused:
            all_experts_count = all_experts_count.sum(axis=-1).reshape([world_size, world_size, 1])
        res = paddle.zeros([world_size, num_expert], dtype=paddle.bool)
        return res

    import os
    if os.environ.get('FMOE_ENABLE_DYNREP', '0') == '1':
        return _global_policy
    else:
        return _global_policy


class HirGate(NaiveGate):
    def __init__(self, d_model, n_expert, world_size, node_rank):
        global cr_cnt
        self.rep_lim = llims[cr_cnt % len(llims)]# 4 - cr_cnt % 4
        cr_cnt += 1
        super().__init__(d_model, n_expert, world_size, topk=2)
        self.ne_per_node = nw_per_node * n_expert
        self.ogn_ratio = .14
        self.node_rank = node_rank

        mask = [0] * world_size * n_expert
        for i in range(n_expert * world_size):
            if i // self.ne_per_node == self.node_rank:
                mask[i] = 1
        self.mask = paddle.to_tensor(mask, dtype='bool')
        self.stored_models = None
        self.policy_fn = _gen_policy(2)

    def forward(self, inp):
        if self.mask.place != inp.place:
            self.mask = paddle.to_tensor(self.mask, place=inp.place)

        gate_score = self.gate(inp)
        lim_mask = self.mask

        # if self.stored_models is not None:
            # lim_mask = lim_mask | self.stored_models.view(-1).to(lim_mask.device)
        lim_mask = ~lim_mask

        top2_val, top2_idx = paddle.topk(gate_score, k=2, axis=-1)
        S = gate_score.shape[0]
        top_k = 2

        with paddle.no_grad():
            top1_idx = top2_idx.reshape((-1, top_k))[:, 0]
            top1_val = top2_val.reshape((-1, top_k))[:, 0]

        c_e = paddle.scatter_nd_add(
                x=paddle.zeros(shape=[self.tot_expert]),
                index=top1_idx.reshape([-1, 1]),
                updates=paddle.ones_like(top1_idx, dtype=paddle.float32),
                ) / S
        
        m_e = paddle.mean(F.softmax(gate_score, axis=1), axis=0)
        loss = paddle.mean(c_e * m_e) * (self.num_expert ** 2)
        self.set_loss(loss)

        with paddle.no_grad():
            #print(top2_idx.dtype)
            _, lec, gec = count_by_gate(top2_idx, 
                    self.num_expert, self.world_size, require_pos=False)
            aec_list = []
            dist.all_gather(aec_list, lec)
            aec = paddle.stack(aec_list)
            agecs = aec.split(num_or_sections=self.world_size, axis=-1)
            #print(agecs)
            agec = paddle.concat(agecs)
            aec = aec.reshape([self.world_size * self.world_size, -1])
            #print(agec)
            stored_models = self.policy_fn(aec, agec,
                    self.num_expert, self.world_size, inp.shape[-1], True)
            # if stored_models.sum().item() < self.rep_lim:
            lim_mask = lim_mask & ~stored_models.reshape([-1])

            # mask for outgoing tokens
            ogn_mask = paddle.cast(lim_mask, dtype=paddle.int32)[top1_idx].cast(dtype=paddle.bool)
            ogn_thres = int(inp.shape[0] * self.ogn_ratio)

        if ogn_mask.sum().item() < ogn_thres:
            topk_val, topk_idx = paddle.topk(gate_score, k=self.top_k)
            topk_val = F.softmax(topk_val, axis=-1)
            return topk_val,topk_idx

        with paddle.no_grad():
            # sys.stderr.write('stored {}\n'.format(self.stored_models))
            # sys.stderr.write('lim_mask {}\n'.format(lim_mask))
            # sys.stderr.write('ogn mask {}\n'.format(ogn_mask))
            # sys.stderr.write('top1 val {}\n'.format(top1_val))
            top1_val[~ogn_mask] = float('-inf')
            _, top_ogn = paddle.topk(top1_val.reshape([-1]), k=ogn_thres)
            cand = gate_score.clone()
            #print(lim_mask)
            cand[:, lim_mask] = float('-inf')
            _, topk_idx = paddle.topk(cand, k=self.top_k)
            # sys.stderr.write(f'{inp.shape}\n')
            # sys.stderr.write(f'{top1_idx.shape}\n')
            # sys.stderr.write(f'{ogn_mask.shape}\n')
            # sys.stderr.write(f'{top_ogn.max()} {top_ogn.shape}\n')
            # sys.stderr.write(f'{topk_idx}\n')
            topk_idx[top_ogn, 1] = top1_idx.reshape([-1])[top_ogn]

        idx_x = paddle.arange(inp.shape[0]).reshape([-1, 1]).tile([1, 2]).reshape([-1])
        #print(idx_x)
        topk_val = gate_score[idx_x, topk_idx.reshape([-1])].reshape([-1, self.top_k])

        # sys.stderr.write('{}: exceeding limits by {} / {}\n'.format(
        #     dist.get_rank(), ogn_mask.sum().item(), ogn_thres))
        # local_expert_count = torch.zeros(
        #     self.num_expert * self.world_size, device=topk_val.device, dtype=torch.int32
        # )
        # fmoe_cuda.expert_count(topk_idx, local_expert_count)
        # local_expert_count = local_expert_count.long().cpu()
        # sys.stderr.write('{}: lec {}\n'.format(dist.get_rank(), local_expert_count))

        # capacity = int(1.2 * inp.shape[0] * self.top_k)
        # _new_lec, _new_gec, topk_idx = limit_by_capacity(
        #         topk_idx, self.num_expert, self.world_size, capacity)

        topk_val = F.softmax(topk_val, axis=-1)
        #print(topk_idx.dtype)
        return topk_val,topk_idx


def gen_hir_gate(rank):
    def _gen(d_model, n_expert, world_size, top_k=2):
        assert top_k == 2
        return HirGate(d_model, n_expert, world_size, rank // nw_per_node)
    return _gen