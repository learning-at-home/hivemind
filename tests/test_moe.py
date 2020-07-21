import numpy as np
import torch
import hivemind
from test_utils.run_server import background_server


def test_moe():
    all_expert_uids = [f'ffn.{np.random.randint(0, 31)}.{np.random.randint(0, 31)}.{np.random.randint(0, 31)}'
                       for _ in range(50)]
    with background_server(expert_uids=all_expert_uids, device='cpu', expert_cls='ffn',
                           num_handlers=1, hidden_dim=16) as (server_endpoint, dht_endpoint):

        dht = hivemind.DHT(start=True, expiration=999, initial_peers=[dht_endpoint])
        # declare expert uids. Server *should* declare them by itself, but it takes time.
        assert all(dht.declare_experts(all_expert_uids, endpoint=server_endpoint))

        dmoe = hivemind.RemoteMixtureOfExperts(
            in_features=16, grid_size=(32, 32, 32), dht=dht, k_best=4, uid_prefix='ffn')

        for i in range(10):
            out = dmoe(torch.randn(10, 16))
            out.sum().backward()


def test_moe_beam_search():
    all_expert_uids = [f'ffn.{5 + i}.{10 + j}.{15 + k}' for i in range(10) for j in range(10) for k in range(10)]
    dht = hivemind.DHT(start=True, expiration=999)
    assert all(dht.declare_experts(all_expert_uids, endpoint='fake-endpoint'))

    dmoe = hivemind.RemoteMixtureOfExperts(
        in_features=32, grid_size=(32, 32, 32), dht=dht, k_best=4, uid_prefix='ffn')

    for i in range(25):
        input = torch.randn(32)
        grid_scores = dmoe.proj(input).split_with_sizes(dmoe.grid_size, dim=-1)

        chosen_experts = dmoe.loop.run_until_complete(dmoe.beam_search(grid_scores, k_best=dmoe.k_best))

        chosen_scores = dmoe.compute_expert_scores([dim_scores[None] for dim_scores in grid_scores],
                                                   [chosen_experts])[0]

        all_scores = dmoe.compute_expert_scores([dim_scores[None] for dim_scores in grid_scores],
                                                [[hivemind.RemoteExpert(uid, '') for uid in all_expert_uids]])[0]
        true_best_scores = sorted(all_scores.cpu().data.numpy(), reverse=True)[:len(chosen_experts)]
        our_best_scores = list(chosen_scores.data.cpu().numpy())
        assert np.allclose(true_best_scores, our_best_scores)


def test_compute_expert_scores():
    try:
        dht = hivemind.DHT(start=True)
        moe = hivemind.client.moe.RemoteMixtureOfExperts(
            dht=dht, in_features=1024, grid_size=(40,), k_best=4, k_min=1, timeout_after_k_min=1,
            uid_prefix='expert')
        gx, gy = torch.randn(4, 5, requires_grad=True), torch.randn(4, 3, requires_grad=True)
        ii = [[4, 0, 2], [3, 1, 1, 1, 3], [0], [3, 2]]
        jj = [[2, 2, 1], [0, 1, 2, 0, 1], [0], [1, 2]]
        batch_experts = [
            [hivemind.RemoteExpert(uid=f'expert.{ii[batch_i][expert_i]}.{jj[batch_i][expert_i]}', endpoint="[::]:1337")
             for expert_i in range(len(ii[batch_i]))]
            for batch_i in range(len(ii))
        ]  # note: these experts do not exists on server, we use them only to test moe compute_expert_scores
        logits = moe.compute_expert_scores([gx, gy], batch_experts)
        torch.softmax(logits, dim=-1).norm(dim=-1).mean().backward()
        assert gx.grad.norm().item() > 0 and gy.grad.norm().item(), "compute_expert_scores didn't backprop"

        for batch_i in range(len(ii)):
            for expert_i in range(len(ii[batch_i])):
                assert torch.allclose(logits[batch_i, expert_i],
                                      gx[batch_i, ii[batch_i][expert_i]] + gy[batch_i, jj[batch_i][expert_i]]), \
                    "compute_expert_scores returned incorrect score"
    finally:
        dht.shutdown()
