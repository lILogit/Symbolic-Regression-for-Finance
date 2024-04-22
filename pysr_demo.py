
#pip3 install -U pysr==0.6.14 pytorch_lightning
#conda install nomkl / solve error
#https://colab.research.google.com/drive/1vgWesrJvsuJR7q-IRjmw2ZzridUTeCDk?usp=sharing#scrollTo=6WuaeqyqbDhe

"""
%%shell
set -e

#---------------------------------------------------#
JULIA_VERSION="1.7.1"
JULIA_PACKAGES="SymbolicRegression"
JULIA_NUM_THREADS=4
#---------------------------------------------------#

if [ -z `which julia` ]; then
  # Install Julia
  JULIA_VER=`cut -d '.' -f -2 <<< "$JULIA_VERSION"`
  echo "Installing Julia $JULIA_VERSION on the current Colab Runtime..."
  BASE_URL="https://julialang-s3.julialang.org/bin/linux/x64"
  URL="$BASE_URL/$JULIA_VER/julia-$JULIA_VERSION-linux-x86_64.tar.gz"
  wget -nv $URL -O /tmp/julia.tar.gz # -nv means "not verbose"
  tar -x -f /tmp/julia.tar.gz -C /usr/local --strip-components 1
  rm /tmp/julia.tar.gz

  for PKG in `echo $JULIA_PACKAGES`; do
    echo "Installing Julia package $PKG..."
    julia -e 'using Pkg; pkg"add '$PKG'; precompile;"'
  done

fi

!pip3 install -U pysr==0.18.2 pytorch_lightning
"""

import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


###### np.random.seed(0)
N = 100000
Nt = 100
X = 6 * np.random.rand(N, Nt, 5) - 3
y_i = X[..., 0] ** 2 + 6 * np.cos(2 * X[..., 2])
y = np.sum(y_i, axis=1) / y_i.shape[1]
z = y**2
X.shape, y.shape

hidden = 100

max_epochs = 1
Xt = torch.tensor(X).float()
zt = torch.tensor(z).float()
hidden = 128
total_steps = 50000

def mlp(size_in, size_out, act=nn.ReLU):
    return nn.Sequential(
        nn.Linear(size_in, hidden),
        act(),
        nn.Linear(hidden, hidden),
        act(),
        nn.Linear(hidden, hidden),
        act(),
        nn.Linear(hidden, size_out),
    )

class SumNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        X_train, X_test, z_train, z_test = train_test_split(Xt, zt, random_state=0)
        self.train_set = TensorDataset(X_train, z_train)
        self.test_set = TensorDataset(X_test, z_test)

        # The same inductive bias as above!
        self.g = mlp(5, 1)
        self.f = mlp(1, 1)

    def forward(self, x):
        y_i = self.g(x)[:, :, 0]
        y = torch.sum(y_i, dim=1, keepdim=True) / y_i.shape[1]
        z = self.f(y)
        return z[:, 0]

    ########################################################

    # PyTorch Lightning bookkeeping:
    def training_step(self, batch, batch_idx):
        x, z = batch
        predicted_z = self(x)
        loss = F.mse_loss(predicted_z, z)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2,
                                                                      steps_per_epoch=len(self.train_dataloader()),
                                                                      epochs=max_epochs,
                                                                      final_div_factor=1e4),
                     'interval': 'step'}
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=128, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=256, num_workers=4)

# ---------- training model Colab / require GPU
# pl.seed_everything(0)
# model = SumNet()
# trainer = pl.Trainer(max_epochs=max_epochs)
# trainer.fit(model)
# trainer.save_checkpoint("example.ckpt")
# ----------

model = SumNet.load_from_checkpoint(checkpoint_path="example.ckpt")

np.random.seed(0)
idx = np.random.randint(0, 10000, size=1000)

X_for_pysr = Xt[idx]
y_i_for_pysr = model.g(X_for_pysr)[:, :, 0]
y_for_pysr = torch.sum(y_i_for_pysr, dim=1) / y_i_for_pysr.shape[1]
z_for_pysr = zt[idx]  # Use true values.

X_for_pysr.shape, y_i_for_pysr.shape

np.random.seed(1)
tmpX = X_for_pysr.detach().numpy().reshape(-1, 5)
tmpy = y_i_for_pysr.detach().numpy().reshape(-1)
idx2 = np.random.randint(0, tmpy.shape[0], size=3000)

model = PySRRegressor(
    niterations=50,
    binary_operators=["plus", "sub", "mult"],
    unary_operators=["cos", "square", "neg"],
)
model.fit(X=tmpX[idx2], y=tmpy[idx2])

#$(0.477 x_{0}^{2} + 2.86 \cos{\left(2 x_{2} \right)} - 0.905)

"""
A neural network can easily undo a linear transform (which commutes with the summation), so any affine transform in 
 is to be expected. The network for 
 has learned to undo the linear transform.

This likely won't find the exact result, but it should find something similar. 
You may wish to try again but with many more total_steps for the neural network (10,000 is quite small!).

Then, we can learn another analytic equation for 
.

Now, we can compose these together to get the time series model!

Think about what we just did: we found an analytical equation for 
 in terms of 
 datapoints, under the assumption that 
 is a function of a sum of another function over an axis:

And we pulled out analytical copies for 
 using symbolic regression.
"""
