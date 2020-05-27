import numpy as np
import torch as th
from tree_text_gen.binary.common.tree import Tree, Node
from tree_text_gen.binary.common.util import pairdist

class Oracle(object):
    def __init__(self, token_idxs, sample_dim, tok2i, i2tok, embeds, greedy=False):
        self.device = token_idxs.device
        self._B = token_idxs.size(0)
        self.tok2i = tok2i
        self.i2tok = i2tok
        self.trees = []
        self.embeds = embeds
        invalid_behavior = 'split'
        for i in range(self._B):
            # Make a tree with only the token indices (not <s>, </s>, <p>)
            filt = (token_idxs[i] != tok2i['</s>']) & (token_idxs[i] != tok2i['<p>']) & (token_idxs[i] != tok2i['<s>'])
            idxs = token_idxs[i][filt].tolist()
            node = Node(idxs, parent=None, end_idx=tok2i['<end>'], invalid_behavior=invalid_behavior)
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
            samples = ps.argmax(1, keepdim=True)
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
                    valid_actions = th.tensor(self.trees[i].current.valid_actions, dtype=th.long, device=self.device)
                    # NOTE(wellecks): uses 1 instead of count (free class vs. free item)
                    self._valid_actions[i].scatter_(0, valid_actions, 1)
        return self._valid_actions.clone()

    def distribution(self):
        # uniform over free labels (in our case, a free class)
        free_labels = self.valid_actions_vector()
        dist = free_labels / th.clamp(free_labels.sum(1, keepdim=True), min=1.0)
        return dist
    
    def choose_action(self, sample, valid_actions):
        #print("VA", valid_actions)
        dists = pairdist(self.embeds[[sample], :], self.embeds[valid_actions, :])
        #print("DISTS", dists)
        element = th.argmin(dists).item()
        reward = -dists[0, element].item()
        actions = valid_actions[element]
        #print("RA", reward, actions)
        return reward, actions
        
    def update(self, samples):
        """Update data structures based on the model's samples."""
        rewards = []
        actions = []
        with th.no_grad():
            for i in range(self._B):
                if self._stopped[i] or len(self.trees[i].current.valid_actions) == 1:
                    # if tree completed and predicted <end> reward is 0 
                    if samples[i].item() == self.end_idx:
                        rewards.append([1.0])
                    else:
                        # if tree completed but predicted not <end> reward is -2
                        rewards.append([-2])
                    actions.append([samples[i]])
                else:
                    valid_actions = self.trees[i].current.valid_actions
                    if samples[i].item() == self.end_idx:
                          rewards.append([(-2) * len(valid_actions)])
                          actions.append([samples[i]])
                    else:
                        reward, action = self.choose_action(samples[i], valid_actions)
                        if reward == 0:
                            reward = 1.0
                        actions.append([action])
                        rewards.append([reward])
        #print(len(rewards))
        return rewards, actions

    def done(self):
        return bool((self._stopped == 1).all().item())


class LeftToRightOracle(Oracle):
    """Only places probability on the 'left-most' valid action."""
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
