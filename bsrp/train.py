"""
bsrp/train.py — laço de treino conforme PROTOCOLO_RIGOR.md §2–§4.

* Aleatoriedade: SeedSequence([seed, SALT]) -> 4 fluxos independentes (init, exploração, ambiente, amostragem do replay).
* Orçamento em PASSOS DE AMBIENTE (decisões), igual para todos os aprendizes.
* Avaliação periódica SÓ em validação (cenários fixos); regra de checkpoint: o ÚLTIMO (sem seleção).
* O teste é tocado uma vez, por `final_eval`, depois do treino.
"""
from __future__ import annotations

import time

import numpy as np

from .agents import DQN, PolicyGradient, QTabular
from .env import BSRPEnv, Instance, ScenarioSet, evaluate, run_episode, sample_scenarios

SALT = 20261012


def streams(seed: int):
    ss = np.random.SeedSequence([seed, SALT])
    return [np.random.default_rng(c) for c in ss.spawn(4)]      # init, explore, env_train, replay_sample


def make_agent(method: str, inst: Instance, seed: int, **kw):
    """method: tab_cluster | tab_station | linear | dqn (rede 64-64)."""
    r_init, _, _, r_samp = streams(seed)
    if method == "tab_cluster":
        return QTabular(inst, "cluster", **kw)
    if method == "tab_station":
        return QTabular(inst, "station", **kw)
    if method == "linear":
        return DQN(inst, hidden=(), rng_init=r_init, rng_sample=r_samp, **kw)
    if method == "dqn":
        return DQN(inst, hidden=(64, 64), rng_init=r_init, rng_sample=r_samp, **kw)
    if method == "pg":
        return PolicyGradient(inst, rng_init=r_init, rng_sample=r_samp, **kw)
    raise ValueError(method)


def train(inst: Instance, agent, seed: int, budget_steps: int, val: ScenarioSet, eval_every: int,
          eps_end: float = 0.05, eps_decay_frac: float = 0.7):
    _, r_explore, r_env, _ = streams(seed)
    env, env_val = BSRPEnv(inst), BSRPEnv(inst)
    tab = isinstance(agent, QTabular)
    decay_steps = max(1, int(eps_decay_frac * budget_steps))
    steps, episodes, next_eval = 0, 0, eval_every
    curve = []
    t0 = time.time()
    while steps < budget_steps:
        sc = sample_scenarios(inst, 1, r_env)[0]
        env.reset(sc)
        while not env.done and steps < budget_steps:
            eps = max(eps_end, 1.0 - (1.0 - eps_end) * steps / decay_steps)
            a = agent.select(env, eps, r_explore)
            if tab:
                k = agent.key(env)
            else:
                s = env.features()
            r, done = env.step(a)
            m2 = env.valid_mask()
            if tab:
                agent.update(k, a, r, agent.key(env), done, m2)
            else:
                agent.observe(s, a, r, env.features(), done, m2)
            steps += 1
        episodes += 1
        if hasattr(agent, "end_episode"):
            agent.end_episode()
        if steps >= next_eval:
            curve.append({"steps": steps, "episodes": episodes, "val": float(evaluate(env_val, agent, val).mean())})
            next_eval += eval_every
    return {"curve": curve, "episodes": episodes, "train_seconds": round(time.time() - t0, 1),
            "n_states": getattr(agent, "n_states", None)}


def final_eval(inst: Instance, agent, test: ScenarioSet) -> dict:
    env = BSRPEnv(inst)
    res = [run_episode(env, agent, test[k]) for k in range(len(test))]
    r = np.array([x["reward"] for x in res])
    return {"test_mean": float(r.mean()), "test_std_scen": float(r.std(ddof=1)),
            "test_unmet_mean": float(np.mean([x["unmet"] for x in res])),
            "test_steps_mean": float(np.mean([x["steps"] for x in res])), "test_rewards": r}
