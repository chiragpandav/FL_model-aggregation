import os
import pickle
from collections import OrderedDict
from typing import Dict, List, OrderedDict, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .util import TEMP_DIR

class Linear(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        nn.init.uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class HyperNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        client_num: int,
        hidden_dim: int,
        backbone: nn.Module,
        K: int,
        gpu=True,
    ):
        super(HyperNetwork, self).__init__()

        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.K = K
        self.client_num = client_num
        self.embedding = nn.Embedding(client_num, embedding_dim, device=self.device)
        self.blocks_name = set(n.split(".")[0] for n, _ in backbone.named_parameters())
        self.cache_dir = TEMP_DIR / "hn"

        # print(" self.cache_dir::  chirag", self.cache_dir)

        if not os.path.isdir(self.cache_dir):
            os.system(f"mkdir -p {self.cache_dir}")

        if os.listdir(self.cache_dir) != client_num:
            for client_id in range(client_num):
                with open(self.cache_dir / f"{client_id}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "mlp": nn.Sequential(
                                nn.Linear(embedding_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                            ),
                            
                            # all negative tensor would be outputted sometimes if fc is torch.nn.Linear, which used kaiming init.
                            # so here use U(0,1) init instead.

                            "fc": {
                                name: Linear(hidden_dim, client_num)
                                for name in self.blocks_name
                            },
                        },
                        f,
                    )

        # for tracking the current client's hn parameters
        self.current_client_id: int = None
        self.mlp: nn.Sequential = None
        self.fc_layers: Dict[str, Linear] = {}
        self.retain_blocks: List[str] = []

    def mlp_parameters(self) -> List[nn.Parameter]:

        # print("self.mlp.parameters():: ", self.mlp)

        return list(filter(lambda p: p.requires_grad, self.mlp.parameters()))

    def fc_layer_parameters(self) -> List[nn.Parameter]:
        params_list = []
        for block, fc in self.fc_layers.items():
            if block not in self.retain_blocks:
                params_list += list(filter(lambda p: p.requires_grad, fc.parameters()))

        return params_list

    def emd_parameters(self) -> List[nn.Parameter]:
        return list(self.embedding.parameters())

    def forward(self, client_id: int) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        self.current_client_id = client_id

        print(" self.current_client_id : ",self.current_client_id,"\n")

        self.retain_blocks = []
        emd = self.embedding(
            torch.tensor(client_id, dtype=torch.long, device=self.device)
        )
        self.load_hn()

        feature = self.mlp(emd)

        print("TEMP_DIR:: ",TEMP_DIR )

        # print(" features : ",feature,"\n")

        alpha = {
            block: F.relu(self.fc_layers[block](feature)) for block in self.blocks_name
        }

        # print("  alpha: ",alpha,"\n")

        default_weight = torch.tensor(
            [i == client_id for i in range(self.client_num)],
            dtype=torch.float,
            device=self.device,
        )

        print(" self.k ", self.K,"\n")
        # i set K =2 in arg.py

        if self.K > 0:  # HeurpFedLA
            
            blocks_name = []
            self_weights = []
            
            with torch.no_grad():
                for name, weight in alpha.items():

                    # print("-: NAME AND WEIGHT : ", name, weight,"\n")

                    blocks_name.append(name)
                    self_weights.append(weight[client_id])

                # not in the Loop
                _, topk_weights_idx = torch.topk(torch.tensor(self_weights), self.K)

                print("  topk_weights_idx ",topk_weights_idx ,"\n")
                
            for i in topk_weights_idx:
                # print(" topk_weights_idx I  ",i,"\n")
                # print(" blocks_name[i] Topk ",blocks_name[i],"\n")
                # print(" default_weight[i] Topk ",default_weight,"\n")

                alpha[blocks_name[i]] = default_weight
                self.retain_blocks.append(blocks_name[i])

        return alpha, self.retain_blocks

    def save_hn(self):
        for block, param in self.fc_layers.items():
            self.fc_layers[block] = param.cpu()
        with open(self.cache_dir / f"{self.current_client_id}.pkl", "wb") as f:
            pickle.dump(
                {"mlp": self.mlp.cpu(), "fc": self.fc_layers}, f,
            )
        self.mlp = None
        self.fc_layers = {}
        self.current_client_id = None

    def load_hn(self) -> Tuple[nn.Sequential, OrderedDict[str, Linear]]:
        with open(self.cache_dir / f"{self.current_client_id}.pkl", "rb") as f:
            parameters = pickle.load(f)
        self.mlp = parameters["mlp"].to(self.device)
        for block, param in parameters["fc"].items():
            self.fc_layers[block] = param.to(self.device)

    def clean_models(self):
        if os.path.isdir(self.cache_dir):
            os.system(f"rm -rf {self.cache_dir}")


# (input_channels, first fc layer's input features, classes)
ARGS = {
    "cifar10": (3, 8192, 10),
    "cifar100": (3, 8192, 100),
    "emnist": (1, 6272, 62),
    "fmnist": (1, 6272, 10),
}


# NOTE: unknown CNN model structure
# Really don't know the specific structure of CNN model used in pFedLA.
# Structures below are from FedBN's.

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 14 : input shape
        self.layer1 = nn.Linear(14, 512)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(256, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x
    

class CNNWithBatchNorm(nn.Module):
    def __init__(self, dataset):
        super(CNNWithBatchNorm, self).__init__()
        self.block1 = nn.ModuleDict(
            {
                "conv": nn.Conv2d(ARGS[dataset][0], 64, 5, 1, 2),
                "bn": nn.BatchNorm2d(64),
                "relu": nn.ReLU(True),
                "pool": nn.MaxPool2d(2),
            }
        )
        self.block2 = nn.ModuleDict(
            {
                "conv": nn.Conv2d(64, 64, 5, 1, 2),
                "bn": nn.BatchNorm2d(64),
                "relu": nn.ReLU(True),
                "pool": nn.MaxPool2d(2),
            }
        )
        self.block3 = nn.ModuleDict(
            {
                "conv": nn.Conv2d(64, 128, 5, 1, 2),
                "bn": nn.BatchNorm2d(128),
                "relu": nn.ReLU(True),
            }
        )
        self.block4 = nn.ModuleDict(
            {"fc": nn.Linear(ARGS[dataset][1], 2048), "relu": nn.ReLU(True)}
        )
        self.block5 = nn.ModuleDict({"fc": nn.Linear(2048, 512), "relu": nn.ReLU(True)})

        self.block6 = nn.ModuleDict({"fc": nn.Linear(512, ARGS[dataset][2])})

    def forward(self, x):
        x = self.block1["conv"](x)
        x = self.block1["bn"](x)
        x = self.block1["relu"](x)
        x = self.block1["pool"](x)

        x = self.block2["conv"](x)
        x = self.block2["bn"](x)
        x = self.block2["relu"](x)
        x = self.block2["pool"](x)

        x = self.block3["conv"](x)
        x = self.block3["bn"](x)
        x = self.block3["relu"](x)

        x = x.view(x.shape[0], -1)

        x = self.block4["fc"](x)
        x = self.block4["relu"](x)

        x = self.block5["fc"](x)
        x = self.block5["relu"](x)

        x = self.block6["fc"](x)
        
        return x

class CNNWithoutBatchNorm(nn.Module):
    def __init__(self, dataset):
        super(CNNWithoutBatchNorm, self).__init__()

        self.block1 = nn.ModuleDict(
            {
                "conv": nn.Conv2d(ARGS[dataset][0], 64, 5, 1, 2),
                "relu": nn.ReLU(True),
                "pool": nn.MaxPool2d(2),
            }
        )
        self.block2 = nn.ModuleDict(
            {
                "conv": nn.Conv2d(64, 64, 5, 1, 2),
                "relu": nn.ReLU(True),
                "pool": nn.MaxPool2d(2),
            }
        )
        self.block3 = nn.ModuleDict(
            {"conv": nn.Conv2d(64, 128, 5, 1, 2), "relu": nn.ReLU(True),}
        )

        self.block4 = nn.ModuleDict(
            {"fc": nn.Linear(ARGS[dataset][1], 2048), "relu": nn.ReLU(True)}
        )
        self.block5 = nn.ModuleDict({"fc": nn.Linear(2048, 512), "relu": nn.ReLU(True)})
        self.block6 = nn.ModuleDict({"fc": nn.Linear(512, ARGS[dataset][2])})

    def forward(self, x):
        x = self.block1["conv"](x)
        x = self.block1["relu"](x)
        x = self.block1["pool"](x)

        x = self.block2["conv"](x)
        x = self.block2["relu"](x)
        x = self.block2["pool"](x)

        x = self.block3["conv"](x)
        x = self.block3["relu"](x)

        x = x.view(x.shape[0], -1)

        x = self.block4["fc"](x)
        x = self.block4["relu"](x)

        x = self.block5["fc"](x)
        x = self.block5["relu"](x)

        x = self.block6["fc"](x)
        return x
