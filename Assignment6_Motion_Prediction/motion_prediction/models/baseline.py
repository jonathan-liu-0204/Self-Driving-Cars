import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from models.subnets.subnets import MultiheadAttention, MLP, MapNet, SubGraph
import yaml 

import math

''' Yaml Parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
_output_heads = int(config['net']['output_heads'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

''' Model
'''
class Baseline(pl.LightningModule):
    def __init__(self):
        super(Baseline, self).__init__()
        ''' history state (x, y, vx, vy, yaw, object_type) * 5s * 10Hz
        '''
        # self.history_encoder = MLP(..., 128, 128)
        self.history_encoder = MLP(300, 128, 128)

        self.lane_encoder = MapNet(2, 128, 128, 10)
        self.lane_attn = MultiheadAttention(128, 8)
        
        trajs = []
        confs = []

        ''' we predict 6 different future trajectories to handle different possible cases.
        '''
        for i in range(6):
            ''' future state (x, y, vx, vy, yaw) * 6s * 10Hz
            '''
            trajs.append(
                #MLP(128, 256, ...)
                MLP(128, 256, 300)
                )
            ''' we use model to predict the confidence score of prediction
            '''
            confs.append(
                    nn.Sequential(
                    MLP(128, 64, 1),
                    nn.Sigmoid()
                    )
                )
        self.future_decoder_traj = nn.ModuleList(trajs)
        self.future_decoder_conf = nn.ModuleList(confs)

    def forward(self, data):
        ''' In deep learning, data['x'] means input, data['y'] means groundtruth
        '''
        # x = data['x'].reshape(-1, ...)
        print("data['x']: {}".format(data['x'].size()))
        x = data['x'].reshape(-1, 300)

        x = self.history_encoder(x)
        	
        lane = data['lane_graph']
        lane = self.lane_encoder(lane)
        
        x = x.unsqueeze(0)
        lane = lane.unsqueeze(0)

        print("x: {}".format(x.size()))

        lane_mask = data['lane_mask']

        print("lane_mask: {}".format(lane_mask.size()))

        lane_attn_out = self.lane_attn(x, lane, lane, attn_mask=lane_mask) 
        
        x = x + lane_attn_out
        x = x.squeeze(0)
        
        trajs = []
        confs = []
        for i in range(6):
            trajs.append(self.future_decoder_traj[i](x))
            confs.append(self.future_decoder_conf[i](x))
        trajs = torch.stack(trajs, 1)
        confs = torch.stack(confs, 1)
        
        return trajs, confs
	