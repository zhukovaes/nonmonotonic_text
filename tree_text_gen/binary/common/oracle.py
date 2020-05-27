import numpy as np
import torch as th
from .tree import Tree, Node


class Oracle(object):
    def __init__(self, token_idxs, sample_dim, tok2i, i2tok, greedy=False, determine=False):
        #print("IT IS NEW COOL ORACLE")
        self.device = token_idxs.device
        self._B = token_idxs.size(0)
        self.tok2i = tok2i
        self.i2tok = i2tok
        self.trees = []
        invalid_behavior = 'split'
        self.determine = determine
        for i in range(self._B):
            # Make a tree with only the token indices (not <s>, </s>, <p>)
            filt = (token_idxs[i] != tok2i['</s>']) & (token_idxs[i] != tok2i['<p>']) & (token_idxs[i] != tok2i['<s>'])
            idxs = token_idxs[i][filt].cpu().numpy()#numpy
            node = Node(idxs, parent=None, end_idx=tok2i['<end>'], invalid_behavior=invalid_behavior, determine=self.determine)
            self.trees.append(Tree(root_node=node, end_idx=tok2i['<end>']))

        # Mask is 1 after generation is complete. B x 1
        self._stopped = th.zeros(self._B).byte()
        self._valid_actions = th.zeros(self._B, sample_dim, requires_grad=False, device=self.device)
        self._sample_dim = sample_dim
        self.end_idx = tok2i['<end>']
        self.greedy = greedy

    def sample(self):
        ps = self.distribution()
        if self.greedy:
            if self.determine:
                # if determine get last value with max probability
                samples = ps.argmax(1, keepdim=True)
            else:
                # sample one of values with max probability
                maxes = ps.max(1, keepdim=True).values
                probs = th.zeros_like(ps, dtype=th.float32)
                probs[th.where(ps == maxes)] = 1.
                samples = probs.multinomial(1)
        else:
            samples = ps.multinomial(1)
        return samples

    def valid_actions_vector(self):
        with th.no_grad():
            self._valid_actions.zero_()
            for i in range(self._B):
                if self._stopped[i]:
                    self._valid_actions[i][self.tok2i['<p>']] = 1
                else:
                    idxes, counts = np.unique(self.trees[i].current.valid_actions, return_counts=True)
                    self._valid_actions[i, idxes] = th.tensor(counts, dtype=th.float32).to(self.device)
        return self._valid_actions.clone()

    def distribution(self):
        # uniform over free items
        free_labels = self.valid_actions_vector()
        dist = free_labels / th.clamp(free_labels.sum(1, keepdim=True), min=1.0)
        return dist

    def update(self, samples):
        """Update data structures based on the model's samples."""
        with th.no_grad():
            for i in range(self._B):
                if not self._stopped[i]:
                    self.trees[i].generate(samples[i].item())
                    self.trees[i].next()
                    self._stopped[i] = int(self.trees[i].done())

    def done(self):
        return bool((self._stopped == 1).all().item())


class LeftToRightOracle(Oracle):
    """Only places probability on the 'left-most' valid action."""
    def __init__(self, token_idxs, sample_dim, tok2i, i2tok):
      super().__init__(token_idxs, sample_dim, tok2i, i2tok, greedy=False, determine=True)

    def valid_actions_vector(self):
        with th.no_grad():
            self._valid_actions.zero_()
            for i in range(self._B):
                if self._stopped[i]:
                    self._valid_actions[i][self.tok2i['<p>']] = 1
                else:
                    # Only retain the 'left-most' valid action
                    left_action = self.trees[i].current.valid_actions[0]
                    self._valid_actions[i, left_action] = 1
        return self._valid_actions.clone().to(self.device)
