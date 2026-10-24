"""可信 CUDA 集成测试；会启动 GPU 3。

运行：PYTHONPATH=src python -B tests/p1/gpu_smoke.py。
输入：固定 fixture、kernel 环境和 Docker 镜像；输出：验收结果与 runs 日志。
不在 unittest discover 中自动执行，当前修改阶段不运行此脚本。"""

import json
from pathlib import Path
from p1.data import ROOT, dump
from p1.evaluation import Evaluator


def main():
    """无输入参数；在 GPU 3 验证可信 CUDA fixture，返回 None；绝不计作模型结果。"""
    cfg = json.loads((ROOT / "configs/p1-pilot.json").read_text())
    cfg.update(timing_iterations=30, timing_rounds=3)
    run = ROOT / "runs/p1/protocol-smoke"
    task = {
        "task_id": "fixture-axpby",
        "code": """import torch


class Model(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return self.alpha * a + b


def get_inputs():
    return [torch.randn(4096), torch.randn(4096)]


def get_init_inputs():
    return [0.25]
""",
    }
    work = run / "workspace"
    (work / "kernels").mkdir(parents=True, exist_ok=True)
    for n in ["binding.cpp", "binding_registry.h"]:
        (work / n).write_bytes((ROOT / "upstream/CUDA-Agent/agent_workdir" / n).read_bytes())
    (work / "model_new.py").write_text(
        """import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return cuda_extension.axpby(a, b, self.alpha)
"""
    )
    (work / "kernels/kernel.cu").write_text(
        """#include <cuda_runtime.h>
__global__ void k(const float*a,const float*b,float*c,int n,float alpha){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)c[i]=alpha*a[i]+b[i];
}
extern "C" void launch(const float*a,const float*b,float*c,int n,float alpha,cudaStream_t s){
    k<<<(n+255)/256,256,0,s>>>(a,b,c,n,alpha);
}
"""
    )
    (work / "kernels/kernel_binding.cpp").write_text(
        """#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
extern "C" void launch(const float*,const float*,float*,int,float,cudaStream_t);
torch::Tensor axpby(torch::Tensor a,torch::Tensor b,float alpha){
    auto c=torch::empty_like(a);
    launch(a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),a.numel(),alpha,c10::cuda::getCurrentCUDAStream().stream());
    return c;
}
void bind(pybind11::module&m){
    m.def("axpby",&axpby);
}
REGISTER_BINDING(axpby,bind);
"""
    )
    ev = Evaluator(cfg, run)
    p = ev.prepare(task)
    print("PREPARED", json.dumps(p), flush=True)
    assert p["status"] == "ok", p
    r = ev.evaluate(task, work)
    print("FEEDBACK", json.dumps(r), flush=True)
    assert r["correct"], r
    r = ev.evaluate(task, work, final=True)
    print("HIDDEN", json.dumps(r), flush=True)
    assert r["correct"], r
    q = work / "kernels/kernel.cu"
    q.write_text(q.read_text().replace("alpha*a[i]+b[i]", "0.0f"))
    r = ev.evaluate(task, work)
    assert not r["correct"], r
    print("PROTOCOL_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
