from typing import List

import numpy as np
import torch
import torch.nn as nn
from hifuku.batch_network import BatchFCN
from hifuku.core import SolutionLibrary


class FeasibilityCheckerBatchImageJit:
    ae_model_shared: torch.jit.ScriptModule

    def __init__(self, lib: SolutionLibrary, n_batch: int):
        self.dummy_encoded = torch.zeros(n_batch, 200).float().cuda()
        self.biases = torch.tensor(lib.biases).float().cuda()
        self.max_admissible_cost = lib.max_admissible_cost

        # ae part
        dummy_input = torch.zeros(n_batch, 1, 112, 112).float().cuda()
        traced = torch.jit.trace(lib.ae_model_shared.eval(), (dummy_input,))
        self.ae_model_shared = torch.jit.optimize_for_inference(traced)

        # vector part
        linear_list = []
        expander_list = []
        for pred in lib.predictors:
            linear_list.append(pred.linears)
            expander_list.append(pred.description_expand_linears)
        fcn_linears_batch = BatchFCN(linear_list).cuda()
        fcn_expanders_batch = BatchFCN(expander_list).cuda()

        class Tmp(nn.Module):
            def __init__(self, fcn_linears, fcn_expanders, n):
                super().__init__()
                self.fcn_linears = fcn_linears
                self.fcn_expanders = fcn_expanders
                self.n = n

            def forward(self, bottlenecks, descriptor):
                # bottlenecks: (n_batch, n_latent)
                n_batch = bottlenecks.shape[0]
                expanded = self.fcn_expanders(descriptor.unsqueeze(0))
                expanded_repeat = expanded.repeat(n_batch, 1, 1)
                bottlenecks_repeat = bottlenecks.unsqueeze(1).repeat(1, self.n, 1)
                concat = torch.cat((bottlenecks_repeat, expanded_repeat), dim=2)
                tmp = self.fcn_linears(concat)
                return tmp.squeeze(2)

            def cuda(self):
                self.fcn_linears.cuda()
                self.fcn_expanders.cuda()
                return super().cuda()

        tmp = Tmp(fcn_linears_batch, fcn_expanders_batch, len(lib.predictors)).cuda()
        dummy_input = (torch.zeros(n_batch, 200).float().cuda(), torch.zeros(7).float().cuda())
        traced = torch.jit.trace(tmp, dummy_input)
        self.batch_predictor = torch.jit.optimize_for_inference(traced)

        # warm up
        vector = np.random.randn(7).astype(np.float32)
        hmaps = [np.random.randn(112, 112).astype(np.float32) for _ in range(n_batch)]
        for _ in range(10):
            self.infer(vector, hmaps)

    def infer(self, vector: np.ndarray, mat_lits: List[np.ndarray]):
        n_batch_actual = len(mat_lits)
        vector = torch.from_numpy(vector).float().cuda()

        mats = torch.stack([torch.from_numpy(mat).float() for mat in mat_lits]).unsqueeze(1).cuda()
        encoded = self.ae_model_shared.forward(mats)
        self.dummy_encoded[:n_batch_actual] = encoded

        costs = self.batch_predictor(self.dummy_encoded, vector)
        cost_calibrated = costs[:n_batch_actual] + self.biases
        min_costs, min_indices = torch.min(cost_calibrated, dim=1)
        return (
            min_costs.cpu().detach().numpy() < self.max_admissible_cost,
            min_indices.cpu().detach().numpy(),
        )

    def infer_single(self, vector: np.ndarray, mat: np.ndarray):
        feasibilities, libtraj_idx = self.infer(vector, [mat])
        return feasibilities[0], libtraj_idx[0]
