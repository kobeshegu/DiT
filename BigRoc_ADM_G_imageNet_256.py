from robustness import model_utils, datasets
import torch

## create Robust classifier
def create_dl_model(DATA='CIFAR', BATCH_SIZE=128, NUM_WORKERS=8):
    '''
    :param DATA: Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
    :param bs: batch size
    :param num_workers:
    :return: a dataloader object
    '''

    # Load dataset
    dataset_function = getattr(datasets, DATA)
    dataset = dataset_function('data')
    # Load model
    model_kwargs = {
        'arch': 'resnet50',
        'dataset': dataset,
        'resume_path': f'./{DATA}.pt'
    }
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    classifier = model.model
    classifier.eval()
    return classifier

adv_model = create_dl_model(DATA='ImageNet')


##
import torch
from torch import nn


class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''

    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x


from torch import nn

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """

    def project(self, x):
        """
        """
        if self.orig_input is None: self.orig_input = x.detach()
        self.orig_input = self.orig_input.cuda()
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        l = len(x.shape) - 1
        g_norm = torch.norm(g.reshape(g.shape[0], -1), dim=1).view(-1, *([1] * l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

def targeted_pgd_l2(model, X, y, num_iter, eps, step_size):
    # input images are in range [0,1]
    steper = L2Step(eps=eps, orig_input=None, step_size=step_size)
    for t in range(num_iter):
        X = X.clone().detach().requires_grad_(True).cuda()
        loss = nn.CrossEntropyLoss(reduction='none')(model(norm(X)), y)
        loss = torch.mean(loss)
        grad, = torch.autograd.grad(-1 * loss, [X])
        X = steper.step(X, grad)
        X = steper.project(X)
    return X.detach()

import numpy as np

x = np.load("/mnt/petrelfs/yangmengping/ckpt/ImageNet/admnet_guided_imagenet256.npz", mmap_mode='r') 
arr_orig = x['arr_0']
gt_labels = x['arr_1']


# BIGRoC args
epsilon, steps = 1, 7
step_size = (epsilon * 1.5) / steps

from tqdm.notebook import tqdm

torch.manual_seed(1234)
for k in range(50):
  boosted_gen_imgs = []
  for i in tqdm(range(100)):
    x_batch = arr_orig[(1000*k) + i * 10: (1000*k) + (i + 1) * 10]
    x_batch_0_1 = torch.tensor(x_batch / 255.).cuda().permute(0,3,1,2).float()
    with torch.no_grad():
      # labels = torch.argmax(adv_model(norm(x_batch_0_1)), dim=1)
      labels = torch.tensor(gt_labels[(1000*k) + i * 10: (1000*k) + (i + 1) * 10]).cuda().long()
    b_imgs = targeted_pgd_l2(model=adv_model, X=x_batch_0_1.data, y=labels.long(), num_iter=steps, eps=epsilon,
                                      step_size=step_size).detach().cpu()
    boosted_gen_imgs.append(b_imgs)

  boosted_gen_imgs = torch.cat(boosted_gen_imgs)
  boosted_gen_imgs = (boosted_gen_imgs * 255.).int()
  boosted_gen_imgs = boosted_gen_imgs.detach().cpu().permute(0,2,3,1).numpy().astype(np.uint8)
  np.savez(f"/mnt/petrelfs/yangmengping/generate_data/ImageNet256/BigRoc/boosted_diff_256_{k}_eps_{epsilon}_steps_{steps}_labeled", boosted_gen_imgs)
  # np.savez(f"./boosted_diff_256_{k}_eps_{epsilon}_steps_{steps}", boosted_gen_imgs)

import numpy as np
from tqdm.notebook import tqdm

arr = np.zeros(shape=(50000, 256, 256, 3), dtype=np.uint8)

for f in tqdm(range(50)):
  x = np.load(f"/mnt/petrelfs/yangmengping/generate_data/ImageNet256/BigRoc/boosted_diff_256_{f}_eps_{epsilon}_steps_{steps}_labeled.npz")['arr_0']
  # x = np.load(f"./boosted_diff_256_{f}_eps_{epsilon}_steps_{steps}.npz")['arr_0']
  arr[1000 * f: 1000 * (f+1)] = x
  x = None

np.savez(f"boosted_diff_256_eps_{epsilon}_steps_{steps}_labeled", arr)
# np.savez(f"boosted_diff_256_eps_{epsilon}_steps_{steps}", arr)