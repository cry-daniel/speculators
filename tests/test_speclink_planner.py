from speculators.speclink.planner import SpeclinkConfig, SpeclinkPlanner


def _candidates():
    return [
        [{"block": 1, "score": 0.9}, {"block": 2, "score": 0.6}, {"block": 3, "score": 0.1}],
        [{"block": 1, "score": 0.4}, {"block": 4, "score": 0.8}, {"block": 5, "score": 0.2}],
        [{"block": 1, "score": 0.3}, {"block": 6, "score": 0.7}, {"block": 7, "score": 0.2}],
    ]


def test_rho_and_risk_are_position_correct():
    cfg = SpeclinkConfig(shared_budget=1, private_max=2, alpha=0, beta=0)
    plan = SpeclinkPlanner(cfg).plan(_candidates(), [0.8, 0.5, 0.25])
    assert plan.rho == [1.0, 0.8, 0.4]
    assert abs(plan.risk[0] - 4 * 0.8 * 0.2) < 1e-12
    assert abs(plan.risk[1] - 0.8 * 4 * 0.5 * 0.5) < 1e-12


def test_shared_blocks_are_present_for_every_token_and_residual_excludes_shared():
    cfg = SpeclinkConfig(shared_budget=1, private_min=1, private_max=1, alpha=0, beta=0)
    plan = SpeclinkPlanner(cfg).plan(_candidates(), [0.8, 0.5, 0.25])
    assert plan.shared_blocks
    shared = set(plan.shared_blocks)
    for residual, final_blocks in zip(
        plan.residual_blocks_per_token,
        plan.final_blocks_per_token,
        strict=True,
    ):
        assert not shared.intersection(residual)
        assert set(final_blocks) == shared.union(residual)


def test_speclink_union_no_larger_than_independent_for_common_budget():
    candidates = _candidates()
    accept_probs = [0.8, 0.5, 0.25]
    independent = SpeclinkPlanner(
        SpeclinkConfig(layout="independent_topk", topk_per_token=2)
    ).plan(candidates, accept_probs)
    speclink = SpeclinkPlanner(
        SpeclinkConfig(
            layout="speclink_prob",
            shared_budget=1,
            private_min=1,
            private_max=1,
            alpha=0,
            beta=0,
        )
    ).plan(candidates, accept_probs)
    assert len(speclink.union_blocks) <= len(independent.union_blocks)


def test_low_rho_suffix_gets_smaller_residual_budget():
    cfg = SpeclinkConfig(
        shared_budget=1,
        private_min=0,
        private_max=8,
        alpha=8,
        beta=0,
    )
    plan = SpeclinkPlanner(cfg).plan(_candidates(), [0.9, 0.9, 0.2])
    assert plan.per_token_budget[-1] < plan.per_token_budget[0]


def test_high_risk_token_triggers_fallback():
    cfg = SpeclinkConfig(
        layout="speclink_prob_fallback",
        shared_budget=1,
        private_max=2,
        risk_threshold=0.5,
        fallback_enabled=True,
    )
    plan = SpeclinkPlanner(cfg).plan(_candidates(), [0.5, 0.9, 0.9])
    assert 0 in plan.fallback_tokens
