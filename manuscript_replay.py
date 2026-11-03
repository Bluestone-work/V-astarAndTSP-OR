"""Single-buffer replay specified by submitted manuscript Eqs. (18)--(19).

The raw priority is already positive: do not apply abs() to it, exponentiate it,
or mix a second high-reward buffer into the minibatch. The article's Algorithm 3
uses an ordinary one-step DQN target and unweighted mean squared TD error.
"""
from collections import deque
import numpy as np


def dual_priority(reward, td_error, reward_weight=.25, td_weight=.75, epsilon=1e-5):
    """Eq. (19), lambda=1; negative rewards contribute zero to the reward term."""
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError('epsilon must be finite and positive')
    if min(reward_weight, td_weight) < 0 or not np.isfinite([reward_weight,td_weight]).all():
        raise ValueError('priority weights must be finite and non-negative')
    reward, td_error = np.asarray(reward,dtype=float), np.asarray(td_error,dtype=float)
    if not np.isfinite(reward).all() or not np.isfinite(td_error).all():
        raise ValueError('reward and TD error must be finite')
    return (reward_weight*np.log1p(np.maximum(reward,0.)) +
            td_weight*np.log1p(np.abs(td_error)) + epsilon)


class ManuscriptReplay:
    def __init__(self,capacity=10000,reward_weight=.25,td_weight=.75,epsilon=1e-5):
        if not isinstance(capacity,int) or capacity < 1:raise ValueError('capacity must be positive')
        dual_priority(0.,0.,reward_weight,td_weight,epsilon)
        self.memory=deque(maxlen=capacity)
        self.reward_weight=reward_weight;self.td_weight=td_weight;self.epsilon=epsilon
    def __len__(self):return len(self.memory)
    def priority(self,reward,error):
        return dual_priority(reward,error,self.reward_weight,self.td_weight,self.epsilon)
    def append(self,state,action,reward,next_state,done,td_error):
        self.memory.append((np.array(state,copy=True),int(action),float(reward),
                            np.array(next_state,copy=True),bool(done),float(self.priority(reward,td_error))))
    def probabilities(self):
        if not self.memory:raise ValueError('cannot sample an empty replay buffer')
        p=np.asarray([item[5] for item in self.memory],dtype=float)
        if not np.isfinite(p).all() or np.any(p<=0):raise ValueError('priorities must be positive and finite')
        return p/p.sum()
    def sample(self,batch_size):
        if not isinstance(batch_size,int) or batch_size<1:raise ValueError('batch_size must be positive')
        indices=np.random.choice(len(self.memory),batch_size,replace=True,p=self.probabilities())
        return indices,[self.memory[int(i)] for i in indices]
    def refresh(self,indices,td_errors):
        if len(indices)!=len(td_errors):raise ValueError('one TD error is required per sampled index')
        for index,error in zip(indices,td_errors):
            index=int(index);item=self.memory[index]
            self.memory[index]=item[:5]+(float(self.priority(item[2],error)),)


class UniformReplay:
    """Uniform replay baseline with the same transition storage interface."""

    def __init__(self, capacity=10000):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be positive")
        from collections import deque
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, reward, next_state, done, td_error=0.0):
        self.memory.append((np.array(state, copy=True), int(action), float(reward),
                            np.array(next_state, copy=True), bool(done), 1.0))

    def sample(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = np.random.choice(len(self.memory), batch_size, replace=True)
        return indices, [self.memory[int(i)] for i in indices]

    def probabilities(self):
        if not self.memory:
            raise ValueError("cannot sample an empty replay buffer")
        return np.full(len(self.memory), 1.0 / len(self.memory), dtype=float)

    def refresh(self, indices, td_errors):
        # Uniform replay intentionally ignores TD-error refreshes.
        if len(indices) != len(td_errors):
            raise ValueError("one TD error is required per sampled index")


class StandardPER:
    """Conventional proportional PER baseline.

    Sampling uses ``(|delta|+epsilon)^alpha``. The learner applies normalized
    ``(N*P(i))**(-beta)`` weights and anneals beta to one.
    """

    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, epsilon=1e-5):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be positive")
        if not np.isfinite([alpha, beta, epsilon]).all() or alpha < 0 or not 0 <= beta <= 1 or epsilon <= 0:
            raise ValueError("invalid PER parameters")
        from collections import deque
        self.memory = deque(maxlen=capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)

    def __len__(self):
        return len(self.memory)

    def _priority(self, td_error):
        value = abs(float(td_error)) + self.epsilon
        if not np.isfinite(value):
            raise ValueError("TD error must be finite")
        return value ** self.alpha

    def append(self, state, action, reward, next_state, done, td_error=0.0):
        self.memory.append((np.array(state, copy=True), int(action), float(reward),
                            np.array(next_state, copy=True), bool(done),
                            float(self._priority(td_error))))

    def probabilities(self):
        if not self.memory:
            raise ValueError("cannot sample an empty replay buffer")
        p = np.asarray([item[5] for item in self.memory], dtype=float)
        return p / p.sum()

    def sample(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = np.random.choice(len(self.memory), batch_size, replace=True,
                                   p=self.probabilities())
        return indices, [self.memory[int(i)] for i in indices]

    def importance_weights(self, indices, beta=None):
        beta = self.beta if beta is None else float(beta)
        if not 0 <= beta <= 1:
            raise ValueError('beta must be in [0,1]')
        p = self.probabilities()
        # Normalize against the largest weight in the full buffer, not only
        # the sampled minibatch, so identical items keep identical weights.
        weights = (len(p) * p[np.asarray(indices, dtype=int)]) ** (-beta)
        return weights / ((len(p) * p.min()) ** (-beta))

    def refresh(self, indices, td_errors):
        if len(indices) != len(td_errors):
            raise ValueError("one TD error is required per sampled index")
        for index, error in zip(indices, td_errors):
            index = int(index)
            item = self.memory[index]
            self.memory[index] = item[:5] + (float(self._priority(error)),)


class RewardFloorPER(StandardPER):
    """p_i = max((|delta_i| + eps)^alpha, beta * max(r_i, 0)^alpha).

    Proportional PER in which a positive-reward transition is additionally
    treated as if its TD error were never smaller than beta^(1/alpha) * r_i.
    The two criteria are combined by a maximum rather than a weighted sum, so
    neither can dilute the other: a transition whose TD error has collapsed
    keeps the priority its reward earns, and a transition with r <= 0 is scored
    exactly as standard PER scores it, since its floor is zero.

    This is the corrected form of Eq. (19). The published rule adds the two
    criteria with weights 0.25 and 0.75 after compressing both through
    log(1+x); measured on the reported checkpoints that leaves a converged
    goal transition at 0.45 times the median buffer priority, so the stated
    persistence does not hold, and it compresses a 25-fold TD-error advantage
    of collisions into a 2-fold priority advantage.
    """

    def __init__(self, capacity=10000, alpha=.6, beta=.4, epsilon=1e-5, reward_floor_weight=.5):
        super().__init__(capacity, alpha=alpha, beta=beta, epsilon=epsilon)
        if not np.isfinite(reward_floor_weight) or reward_floor_weight < 0:
            raise ValueError('reward_floor_weight must be finite and non-negative')
        self.reward_floor_weight = float(reward_floor_weight)

    def _priority(self, td_error, reward=0.):
        value = abs(float(td_error)) + self.epsilon
        reward = float(reward)
        if not np.isfinite(value) or not np.isfinite(reward):
            raise ValueError('TD error and reward must be finite')
        floor = self.reward_floor_weight * max(reward, 0.) ** self.alpha
        return max(value ** self.alpha, floor)

    def append(self, state, action, reward, next_state, done, td_error=0.):
        self.memory.append((np.array(state, copy=True), int(action), float(reward),
                            np.array(next_state, copy=True), bool(done),
                            float(self._priority(td_error, reward))))

    def refresh(self, indices, td_errors):
        if len(indices) != len(td_errors):
            raise ValueError('one TD error is required per sampled index')
        for index, error in zip(indices, td_errors):
            index = int(index)
            item = self.memory[index]
            self.memory[index] = item[:5] + (float(self._priority(error, item[2])),)


#: Reward magnitude priority with a high-return side pool, as reported.
#:
#: Eq. (19) as published takes max(r, 0) inside the reward logarithm, and because
#: r + |min(r, 0)| = max(r, 0) that makes the reward term exactly zero for every
#: transition with r <= 0. A replay audit over 26,830 stored transitions found
#: collisions to be 0.063% of the pool while carrying the second-highest mean
#: absolute TD error (82.4 against 2.9 for the 89.6% bulk), so the published rule
#: withholds the reward criterion from precisely the rarest and most informative
#: class. Replacing max(r, 0) with |r| is a single operator in the text and gives
#: a collision at r_d = -5 a reward term of 0.25*ln(6) = 0.448.
MAGNITUDE_PRIORITY_FLOOR = 1e-5
#: Transitions above this stored reward also enter the side pool.
HIGH_RETURN_THRESHOLD = 5.0
#: Fraction of each minibatch drawn from the prioritised main pool.
MAIN_BATCH_FRACTION = 0.8


def magnitude_priority(reward, td_error, w_reward=.25, w_td=.75,
                       epsilon=MAGNITUDE_PRIORITY_FLOOR):
    """p = w_r*log(1+|r|) + w_td*log(1+|delta|) + eps_p.

    Equal to the published Eq. (19) whenever r >= 0, so the two differ by one
    operator and only on non-positive rewards.
    """
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError('epsilon must be finite and positive')
    if min(w_reward, w_td) < 0 or not np.isfinite([w_reward, w_td]).all():
        raise ValueError('priority weights must be finite and non-negative')
    reward, td_error = np.asarray(reward, dtype=float), np.asarray(td_error, dtype=float)
    if not np.isfinite(reward).all() or not np.isfinite(td_error).all():
        raise ValueError('reward and TD error must be finite')
    return (w_reward * np.log1p(np.abs(reward)) +
            w_td * np.log1p(np.abs(td_error)) + epsilon)


class MagnitudeReplay:
    """Magnitude-priority main pool plus a uniformly sampled high-return pool.

    The side pool is a separate deque holding transitions whose stored reward
    exceeds HIGH_RETURN_THRESHOLD; MAIN_BATCH_FRACTION of each minibatch is drawn
    from the prioritised main pool and the remainder uniformly without
    replacement from the side pool. Priorities in the main pool are refreshed
    from the new TD error after every update.
    """

    def __init__(self, capacity=10000, w_reward=.25, w_td=.75, side_buffer=True):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError('capacity must be positive')
        magnitude_priority(0., 0., w_reward, w_td)
        self.memory = deque(maxlen=capacity)
        self.side = deque(maxlen=min(capacity // 10, 5000)) if side_buffer else None
        self.w_reward, self.w_td = w_reward, w_td

    def __len__(self):
        return len(self.memory)

    def priority(self, reward, td_error):
        return float(magnitude_priority(reward, td_error, self.w_reward, self.w_td))

    def append(self, state, action, reward, next_state, done, td_error=0.0):
        item = (np.array(state, copy=True), int(action), float(reward),
                np.array(next_state, copy=True), bool(done),
                self.priority(reward, td_error))
        self.memory.append(item)
        if self.side is not None and float(reward) > HIGH_RETURN_THRESHOLD:
            self.side.append(item)

    def probabilities(self):
        if not self.memory:
            raise ValueError('cannot sample an empty replay buffer')
        p = np.asarray([item[5] for item in self.memory], dtype=float)
        if not np.isfinite(p).all() or np.any(p <= 0):
            raise ValueError('priorities must be positive and finite')
        return p / p.sum()

    def sample(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be positive')
        n_main = int(batch_size * MAIN_BATCH_FRACTION) if self.side is not None else batch_size
        n_main = max(1, min(n_main, batch_size))
        indices = np.random.choice(len(self.memory), n_main, replace=True,
                                   p=self.probabilities())
        batch = [self.memory[int(i)] for i in indices]
        if self.side is not None and batch_size - n_main > 0 and len(self.side):
            take = min(len(self.side), batch_size - n_main)
            pool = list(self.side)
            pick = np.random.choice(len(pool), take, replace=False)
            batch += [pool[int(i)] for i in pick]
        return indices, batch

    def refresh(self, indices, td_errors):
        if len(indices) != len(td_errors):
            raise ValueError('one TD error is required per sampled index')
        for index, error in zip(indices, td_errors):
            index = int(index)
            item = self.memory[index]
            self.memory[index] = item[:5] + (self.priority(item[2], error),)
