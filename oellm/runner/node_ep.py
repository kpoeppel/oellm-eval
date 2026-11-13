"""Node-local EP for lm_eval_queue.py: vLLM's DP frontends (one process per DP rank, each a vLLM
DP frontend of one expert-parallel engine) plus lm-eval's distributed request splitting over gloo."""
import multiprocessing as mp
import os
import socket
import time


def _worker(rank, size, evaluator_port, engine_port, argv):
    os.environ.update(OELLM_DP_RANK=str(rank), VLLM_DP_RANK=str(rank),
                      VLLM_DP_RANK_LOCAL=str(rank), VLLM_DP_SIZE=str(size),
                      VLLM_DP_MASTER_IP='127.0.0.1', VLLM_DP_MASTER_PORT=str(engine_port))
    import torch.distributed as dist
    from datetime import timedelta
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{evaluator_port}',
                            rank=rank,world_size=size,timeout=timedelta(minutes=30))
    try:
        from oellm.runner.lm_eval_queue import main
        main(argv)
    finally:
        dist.destroy_process_group()


def launch(size, argv):
    # Reserve both ports together so they cannot accidentally coincide.
    with socket.socket() as one, socket.socket() as two:
        one.bind(('127.0.0.1',0));two.bind(('127.0.0.1',0))
        evaluator_port=one.getsockname()[1];engine_port=two.getsockname()[1]
    ctx=mp.get_context('spawn')
    processes=[ctx.Process(target=_worker,args=(r,size,evaluator_port,engine_port,argv)) for r in range(size)]
    for process in processes:process.start()
    while any(p.is_alive() for p in processes):
        failed=[p for p in processes if p.exitcode not in (None,0)]
        if failed:
            for process in processes:
                if process.is_alive():process.terminate()
            for process in processes:process.join(timeout=5)
            raise RuntimeError(f'EP frontend failure: {[(p.pid,p.exitcode) for p in failed]}')
        time.sleep(0.2)
    for process in processes:process.join()
    if any(p.exitcode for p in processes):raise RuntimeError('EP frontend failed')


def dp_all(flag: bool) -> bool:
    """True on every rank iff ``flag`` is on every rank of the node-local EP frontends (``flag`` itself
    outside them). The frontends drive vLLM's offline DP engines, which have no DP coordinator: an
    engine with requests steps in lockstep with its peers, and a peer whose frontend sends nothing
    stays paused, so the busy one hangs in its DP all-reduce (30 min gloo timeout). Every rank must
    therefore call the engine whenever one does; lm-eval pads the ranks to equal request counts, and
    the caches decide with this collective instead of per rank."""
    if "OELLM_DP_RANK" not in os.environ:
        return flag
    import torch
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return flag
    t = torch.tensor([1 if flag else 0], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


class _CacheMiss:
    """lm-eval's response cache seen as empty for one call; new results are still stored."""

    def __init__(self, real):
        self.real = real

    def __contains__(self, key):
        return False

    def __setitem__(self, key, value):
        self.real[key] = value

    def commit(self):
        self.real.commit()


def consistent_response_cache():
    """lm-eval's CachingLM per DP rank skipped the model for cached requests: a rank with every
    request cached made no engine call, a partly cached one fewer calls than its peers, and the
    engines hung (belebele_hun_Latn retried after a crash, MN5 2026-11-13). Now a request method
    reads the cache only when every rank has all of its requests cached, else every rank
    evaluates all of its requests (equal counts, as lm-eval padded them) and refreshes the cache."""
    from lm_eval.api import model as lm_model

    if getattr(lm_model.CachingLM, "_oellm_dp_consistent", False):
        return
    original = lm_model.CachingLM.__getattr__

    def __getattr__(self, attr):
        fn = original(self, attr)
        if attr not in ("loglikelihood", "loglikelihood_rolling", "generate_until"):
            return fn

        def _fn(requests):
            cached = all(
                not (attr == "generate_until" and r.args[1].get("do_sample", False))
                and lm_model.hash_args(attr, r.args) in self.dbdict for r in requests)
            if dp_all(cached):
                return fn(requests)
            real = self.dbdict
            self.dbdict = _CacheMiss(real)
            try:
                return fn(requests)
            finally:
                self.dbdict = real

        return _fn

    lm_model.CachingLM.__getattr__ = __getattr__
    lm_model.CachingLM._oellm_dp_consistent = True


def distributed_adapter(base):
    class NodeEPVLLM(base):
        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            # the ranks' documents give different window counts, hence different numbers of engine
            # calls: the offline DP engines would hang (dp_all); run such rows with --dp 1
            raise NotImplementedError("loglikelihood_rolling under node-local DP: use --dp 1")

        def all_gather(self, tensor):
            import torch
            import torch.distributed as dist
            output=[torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(output,tensor)
            return torch.cat([x.reshape(1) if x.ndim==0 else x for x in output])

        def gather_object(self, obj, dst=0):
            import torch.distributed as dist
            output=[None]*self.world_size if self.rank==dst else None
            dist.gather_object(obj,output,dst=dst)
            return output

        def barrier(self):
            import torch.distributed as dist
            dist.barrier()

    return NodeEPVLLM
