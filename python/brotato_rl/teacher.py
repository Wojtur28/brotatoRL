"""The bot as a teacher during PPO (kickstarting, an online form of DAgger).

Behaviour cloning (bc.py) copies the bot only on the states the bot itself visits. The cloned
network makes small mistakes, ends up where the bot never was (surrounded in the middle of the
arena on wave 8, say) and does not know how to get out. HeuristicBot is a pure function of the
observation (~0.2 ms per step), so it can label the states the network visits itself. After every
PPO update this callback takes the rollout collected before it, asks the bot what it would have
done in each of those states and runs a pass of cross-entropy towards those actions (policy head
only, its own Adam). The pull decays to 0, after which training is plain PPO.

Config (train.teacher in the training config):
    {"bot_params": {}, "weight": 1.0, "decay_steps": 5000000, "learning_rate": 1e-4,
     "epochs": 1, "batch_size": 512}
Step size = learning_rate * weight(t), weight(t) = weight * (1 - steps / decay_steps), floored at
0: Adam ignores the scale of a loss, so the weight scales the learning rate, not the loss.

Logged every rollout: teacher/weight, teacher/agreement (argmax of the policy that collected the
rollout = the bot's action), teacher/agreement_sampled (the action actually taken = the bot's) and
teacher/cross_entropy (mean of the pass over the previous rollout).
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .bots import HeuristicBot


def teacher_weight(cfg: dict, steps: int) -> float:
    weight, horizon = float(cfg.get("weight", 1.0)), float(cfg.get("decay_steps", 5_000_000))
    if horizon <= 0:
        return weight
    return max(0.0, weight * (1.0 - steps / horizon))


class TeacherCallback(BaseCallback):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.bot = HeuristicBot(**self.cfg.get("bot_params", {}))
        self.optimizer = None
        self.pending = None  # (observations, bot actions) of the last rollout, distilled after PPO's update
        self.rng = np.random.default_rng(0)

    def _on_training_start(self) -> None:
        import torch as th
        self.optimizer = th.optim.Adam(self.model.policy.parameters(), lr=float(self.cfg.get("learning_rate", 1e-4)))

    def _on_step(self) -> bool:
        return True

    def _logits(self, obs, idx):
        import torch as th
        o = {k: th.as_tensor(v[idx], device=self.model.device) for k, v in obs.items()}
        return self.model.policy.get_distribution(o).distribution.logits

    def _on_rollout_end(self) -> None:
        import torch as th
        weight = teacher_weight(self.cfg, self.num_timesteps)
        buf = self.model.rollout_buffer
        obs = {k: v.reshape(-1, *v.shape[2:]).copy() for k, v in buf.observations.items()}
        n = len(buf.actions.reshape(-1))
        labels = np.fromiter((self.bot.act({k: v[i] for k, v in obs.items()}) for i in range(n)),
                             dtype=np.int64, count=n)
        with th.no_grad():
            argmax = np.concatenate([self._logits(obs, np.arange(i, min(i + 2048, n))).argmax(dim=1).cpu().numpy()
                                     for i in range(0, n, 2048)])
        self.logger.record("teacher/weight", weight)
        self.logger.record("teacher/agreement", float(np.mean(argmax == labels)))
        self.logger.record("teacher/agreement_sampled", float(np.mean(buf.actions.reshape(-1).astype(np.int64) == labels)))
        self.pending = (obs, labels) if weight > 0 else None

    def _on_rollout_start(self) -> None:
        # Runs after PPO's update on the previous rollout, so the next rollout is collected (and its
        # old log-probabilities stored) by the policy that already includes this pass.
        if self.pending is None:
            return
        import torch as th
        import torch.nn.functional as F
        obs, labels = self.pending
        self.pending = None
        lr = float(self.cfg.get("learning_rate", 1e-4)) * teacher_weight(self.cfg, self.num_timesteps)
        if lr <= 0:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        policy = self.model.policy
        policy.set_training_mode(True)
        batch, losses = int(self.cfg.get("batch_size", 512)), []
        for _ in range(int(self.cfg.get("epochs", 1))):
            order = self.rng.permutation(len(labels))
            for i in range(0, len(order), batch):
                idx = order[i:i + batch]
                loss = F.cross_entropy(self._logits(obs, idx), th.as_tensor(labels[idx], device=self.model.device))
                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(policy.parameters(), float(self.model.max_grad_norm))
                self.optimizer.step()
                losses.append(float(loss))
        policy.set_training_mode(False)
        self.logger.record("teacher/cross_entropy", float(np.mean(losses)))
