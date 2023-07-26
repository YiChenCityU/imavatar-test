"""
The code is based on https://github.com/lioryariv/idr and https://github.com/xuchen-ethz/snarf
# Copyright (c) ETH Zurich and its affiliates. All rights reserved.
# licensed under the LICENSE file in the root directory of this source tree.
# Code adapted / written by Yufeng Zheng.
"""
import torch
import torch.nn as nn
from utils import rend_util
from utils import general as utils
from utils import plots as plot
from model.ray_tracing import RayTracing
from flame.FLAME import FLAME
from pytorch3d import ops
from functools import partial
#from model.geometry_network import GeometryNetwork
from model.texture_network import RenderingNetwork
from model.deformer_network import ForwardDeformer
from model.geometry_tcnn import GeometryNetwork
import time
from torch import autograd
from . import grid
import numpy as np
print_flushed = partial(print, flush=True)
import torch.nn.functional as F


class IMavatar(nn.Module):
    def __init__(self, conf, shape_params, gt_w_seg):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        
        self.FLAMEServer = FLAME('./flame/FLAME2020/generic_model.pkl', n_shape=100,
                                 n_exp=conf.get_config('deformer_network').get_int('num_exp'),
                                 shape_params=shape_params).cuda()
        self.FLAMEServer.canonical_verts, self.FLAMEServer.canonical_pose_feature, self.FLAMEServer.canonical_transformations = \
            self.FLAMEServer(expression_params=self.FLAMEServer.canonical_exp, full_pose=self.FLAMEServer.canonical_pose)
        self.FLAMEServer.canonical_verts = self.FLAMEServer.canonical_verts.squeeze(0)

        #self.geometry_network = GeometryNetwork(self.feature_vector_size, **conf.get_config('geometry_network'))
        # self.geometry_network = GeometryNetwork(self.feature_vector_size, **conf.get_config('geometry_network_hash'))
        self.world_size = torch.tensor([256, 256, 256])
        bound_size = 1.5
        self.xyz_min = torch.tensor([-bound_size, -bound_size, -bound_size]).cuda()
        self.xyz_max = torch.tensor([bound_size, bound_size, bound_size]).cuda()
        self._set_grid_resolution(256**3)
        self.nearest = False

        self.sdf = grid.create_grid(
            'DenseGrid', channels=1, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        self.color_grid = grid.create_grid(
            'DenseGrid', channels=32, world_size=self.world_size,
            xyz_min=self.xyz_min, xyz_max=self.xyz_max)
        x, y, z = np.mgrid[-1.0:1.0:self.world_size[0].item() * 1j, -1.0:1.0:self.world_size[1].item() * 1j, -1.0:1.0:self.world_size[2].item() * 1j]
        self.sdf.grid.data = torch.from_numpy((x ** 2 + y ** 2 + z ** 2) ** 0.5 -0.4).float()[None, None, ...]
        self.init_gradient_conv()
        self.init_smooth_conv(5, 1)
        self.grad_mode = 'grad_conv'

        self.deformer_class = conf.get_string('deformer_class').split('.')[-1]
        print('deformer creation')
        self.deformer_network = utils.get_class(conf.get_string('deformer_class'))(FLAMEServer=self.FLAMEServer, **conf.get_config('deformer_network'))

        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

        self.ghostbone = self.deformer_network.ghostbone
        if self.ghostbone:
            self.FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), self.FLAMEServer.canonical_transformations], 1)
        self.gt_w_seg = gt_w_seg
        print('Finish init ')

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        #self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        print('voxel_size      ', self.voxel_size)
        #print('world_size      ', self.world_size)

    def init_gradient_conv(self, sigma = 0):
        self.grad_conv = nn.Conv3d(1,3,(3,3,3),stride=(1,1,1), padding=(1, 1, 1), padding_mode='replicate')
        kernel = np.asarray([
            [[1,2,1],[2,4,2],[1,2,1]],
            [[2,4,2],[4,8,4],[2,4,2]],
            [[1,2,1],[2,4,2],[1,2,1]],
        ])
        # sigma controls the difference between naive [-1,1] and sobel kernel
        distance = np.zeros((3,3,3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    distance[i,j,k] = ((i-1)**2 + (j-1)**2 + (k-1)**2 - 1)
        kernel0 = kernel * np.exp(-distance * sigma)

        kernel1 = kernel0 / ( kernel0[0].sum() * 2 * self.voxel_size.item())
        weight = torch.from_numpy(np.concatenate([kernel1[None] for _ in range(3)])).float()
        weight[0,1,:,:] *= 0
        weight[0,0,:,:] *= -1
        weight[1,:,1,:] *= 0
        weight[1,:,0,:] *= -1
        weight[2,:,:,1] *= 0
        weight[2,:,:,0] *= -1
        self.grad_conv.weight.data = weight.unsqueeze(1).float()
        self.grad_conv.bias.data = torch.zeros(3)
        for param in self.grad_conv.parameters():
            param.requires_grad = False

        # smooth conv for TV
        self.tv_smooth_conv = nn.Conv3d(1, 1, (3, 3, 3), stride=1, padding=1, padding_mode='replicate')
        weight = torch.from_numpy(kernel0 / kernel0.sum()).float()
        self.tv_smooth_conv.weight.data = weight.unsqueeze(0).unsqueeze(0).float()
        self.tv_smooth_conv.bias.data = torch.zeros(1)
        for param in self.tv_smooth_conv.parameters():
            param.requires_grad = False
    def _gaussian_3dconv(self, ksize=3, sigma=1):
        x = np.arange(-(ksize//2),ksize//2 + 1,1)
        y = np.arange(-(ksize//2),ksize//2 + 1,1)
        z = np.arange(-(ksize//2),ksize//2 + 1,1)
        xx, yy, zz = np.meshgrid(x,y,z)
        kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
        kernel = torch.from_numpy(kernel).to(self.sdf.grid)
        m = nn.Conv3d(1,1,ksize,stride=1,padding=ksize//2, padding_mode='replicate')
        m.weight.data = kernel[None, None, ...] / kernel.sum()
        m.bias.data = torch.zeros(1)
        for param in m.parameters():
            param.requires_grad = False
        return m

    def init_smooth_conv(self, ksize=3, sigma=1):
        self.smooth_sdf = ksize > 0
        if self.smooth_sdf:
            self.smooth_conv = self._gaussian_3dconv(ksize, sigma)
            print("- "*10 + "init smooth conv with ksize={} and sigma={}".format(ksize, sigma) + " -"*10)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True, smooth=False, displace=0.):
        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3).to(self.xyz_min.device)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if displace !=0:
            ind_norm[...,:] += displace * self.voxel_size
        # TODO: use `rearrange' to make it readable
        if smooth:
            grid = self.smooth_conv(grids[0])
        else:
            grid = grids[0]
        ret_lst = F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners
                                ).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
        return ret_lst

    def neus_sdf_gradient(self, mode=None, sdf=None):
        if sdf is None:
            sdf = self.sdf.grid
        if mode is None:
            mode = self.grad_mode
        if mode == 'interpolate':
            gradient = torch.zeros([1, 3] + [*self.sdf.grid.shape[-3:]]).to(self.sdf.grid.device)
            gradient[:,0,1:-1,:,:] = (sdf[:,0,2:,:,:] - sdf[:,0,:-2,:,:]) / 2 / self.voxel_size
            gradient[:,1,:,1:-1,:] = (sdf[:,0,:,2:,:] - sdf[:,0,:,:-2,:]) / 2 / self.voxel_size
            gradient[:,2,:,:,1:-1] = (sdf[:,0,:,:,2:] - sdf[:,0,:,:,:-2]) / 2 / self.voxel_size
        elif mode == 'grad_conv':
            # use sobel operator for gradient seems basically the same as the naive solution
            for param in self.grad_conv.parameters():
                assert not param.requires_grad
                pass
            gradient = self.grad_conv(sdf)
        elif mode == 'raw':
            gradient = torch.zeros([1, 3] + [*self.sdf.grid.shape[-3:]]).to(self.sdf.grid.device)
            gradient[:,0,:-1,:,:] = (sdf[:,0,1:,:,:] - sdf[:,0,:-1,:,:]) / self.voxel_size
            gradient[:,1,:,:-1,:] = (sdf[:,0,:,1:,:] - sdf[:,0,:,:-1,:]) / self.voxel_size
            gradient[:,2,:,:,:-1] = (sdf[:,0,:,:,1:] - sdf[:,0,:,:,:-1]) / self.voxel_size
        else:
            raise NotImplementedError
        return gradient

    def total_variation(self, v, mask=None):
        assert isinstance(v, torch.Tensor), f"Expected torch.Tensor, but got {type(v)}"
        if torch.__version__ == '1.10.0':
            tv2 = v.diff(dim=2).abs()
            tv3 = v.diff(dim=3).abs()
            tv4 = v.diff(dim=4).abs()
        else:
            tv2 = (v[:,:,1:,:,:] - v[:,:,:-1,:,:]).abs()
            tv3 = (v[:,:,:,1:,:] - v[:,:,:,:-1,:]).abs()
            tv4 = (v[:,:,:,:,1:] - v[:,:,:,:,:-1]).abs()
        if mask is not None:
            tv2 = tv2[mask[:,:,:-1] & mask[:,:,1:]]
            tv3 = tv3[mask[:,:,:,:-1] & mask[:,:,:,1:]]
            tv4 = tv4[mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]]
        return (tv2.sum() + tv3.sum() + tv4.sum()) / 3

    def density_total_variation(self, sdf_tv=0.1, smooth_grad_tv=0.05):
        tv = 0
        assert isinstance(self.sdf.grid, torch.Tensor), f"Expected torch.Tensor, but got {type(self.sdf.grid)}"
        if sdf_tv > 0:
            tv += self.total_variation(self.sdf.grid) / 2 / self.voxel_size * sdf_tv
        if smooth_grad_tv > 0:
            smooth_tv_error = (self.tv_smooth_conv(self.gradient.permute(1,0,2,3,4)).detach() - self.gradient.permute(1,0,2,3,4))
            smooth_tv_error = smooth_tv_error ** 2
            tv += smooth_tv_error.mean() * smooth_grad_tv
        return tv

    def k0_total_variation(self, k0_tv=1., k0_grad_tv=0.):
        v = torch.sigmoid(self.color_grid.grid)
        tv = 0
        if k0_tv > 0:
            assert isinstance(self.sdf.grid, torch.Tensor), f"Expected torch.Tensor, but got {type(self.sdf.grid)}"
            tv += self.total_variation(v)
        if k0_grad_tv > 0:
            raise NotImplementedError
        return tv

    def sample_sdfs(self, xyz, *grids, displace_list, mode='bilinear', align_corners=True, use_grad_norm=False):

        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)

        grid = grids[0]
        # ind from xyz to zyx !!!!!
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        grid_size = grid.size()[-3:]
        size_factor_zyx = torch.tensor([grid_size[2], grid_size[1], grid_size[0]]).cuda()
        ind = ((ind_norm + 1) / 2) * (size_factor_zyx - 1)
        offset = torch.tensor([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]).cuda()
        displace = torch.tensor(displace_list).cuda()
        offset = offset[:, None, :] * displace[None, :, None]

        all_ind = ind.unsqueeze(-2) + offset.view(-1, 3)
        all_ind = all_ind.view(1, 1, 1, -1, 3)
        all_ind[..., 0] = all_ind[..., 0].clamp(min=0, max=size_factor_zyx[0] - 1)
        all_ind[..., 1] = all_ind[..., 1].clamp(min=0, max=size_factor_zyx[1] - 1)
        all_ind[..., 2] = all_ind[..., 2].clamp(min=0, max=size_factor_zyx[2] - 1)

        all_ind_norm = (all_ind / (size_factor_zyx-1)) * 2 - 1
        feat = F.grid_sample(grid, all_ind_norm, mode=mode, align_corners=align_corners)

        all_ind = all_ind.view(1, 1, 1, -1, 6, len(displace_list), 3)
        diff = all_ind[:, :, :, :, 1::2, :, :] - all_ind[:, :, :, :, 0::2, :, :]
        diff, _ = diff.max(dim=-1)
        feat_ = feat.view(1, 1, 1, -1, 6, len(displace_list))
        feat_diff = feat_[:, :, :, :, 1::2, :] - feat_[:, :, :, :, 0::2, :]
        grad = feat_diff / diff / self.voxel_size

        feat = feat.view(shape[-1], 6, len(displace_list))
        grad = grad.view(shape[-1], 3, len(displace_list))

        if use_grad_norm:
            grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-5)

        feat = feat.view(shape[-1], 6 * len(displace_list))
        grad = grad.view(shape[-1], 3 * len(displace_list))

        return feat, grad

    def init_smooth_conv_test_k3(self, ksize=3, sigma=0.4):
        self.smooth_conv_test_k3 = self._gaussian_3dconv(ksize, sigma)
        print("- "*10 + "init smooth conv test with ksize={} and sigma={}".format(ksize, sigma) + " -"*10)

    def query_sdf(self, pnts_p, idx, network_condition, pose_feature, betas, transformations):
        if network_condition is not None:
            network_condition = network_condition[idx]
        pose_feature = pose_feature[idx]
        betas = betas[idx]
        transformations = transformations[idx]
        # print('pnts_p before deform', pnts_p)
        pnts_c, others = self.deformer_network(pnts_p, pose_feature, betas, transformations)
        print('pnts_c shape after deformer : ', pnts_c.shape)
        # print('pnts_p after deform', pnts_c)

        # pnts_c_min = pnts_c.min()
        # pnts_c_max = pnts_c.max()
        #
        # # 将张量归一化到 [0, 1]
        # pnts_c = (pnts_c - pnts_c_min) / (pnts_c_max - pnts_c_min)
        #
        # # 将张量归一化到 [-1, 1]
        # pnts_c = pnts_c * 2 - 1

        num_point, num_init, num_dim = pnts_c.shape
        pnts_c = pnts_c.reshape(num_point * num_init, num_dim)
        # geometry net比deformer net耗时多17倍左右
        # output = self.geometry_network(pnts_c, network_condition).reshape(num_point, num_init, -1)
        # sdf = output[:, :, 0]
        # feature = output[:, :, 1:]
        sdf = self.grid_sampler(pnts_c, self.sdf.grid).reshape(num_point, num_init, -1).squeeze(-1)
        print('sdf shape:',sdf.shape)
        feature = self.grid_sampler(pnts_c, self.color_grid.grid).reshape(num_point, num_init, -1)

        # aggregate occupancy probablities
        mask = others['valid_ids']
        print('mask shape: ', mask.shape)
        sdf[~mask] = 1.
        sdf, index = torch.min(sdf, dim=1)
        print('index shape: ', index.shape)
        pnts_c = pnts_c.reshape(num_point, num_init, num_dim)

        pnts_c = torch.gather(pnts_c, dim=1, index=index.unsqueeze(-1).unsqueeze(-1).expand(num_point, num_init, num_dim))[:, 0, :]
        feature = torch.gather(feature, dim=1, index=index.unsqueeze(-1).unsqueeze(-1).expand(num_point, num_init, feature.shape[-1]))[:, 0, :]
        mask = torch.gather(mask, dim=1, index=index.unsqueeze(-1).expand(num_point, num_init))[:, 0]

        return sdf, pnts_c, feature, {'mask': mask}

    def forward(self, input, return_sdf=False):

        uv = input["uv"]
        intrinsics = input["intrinsics"]
        cam_pose = input["cam_pose"]
        flame_pose = input["flame_pose"]
        expression = input["expression"]
        object_mask = input["object_mask"].reshape(-1) if "object_mask" in input else None
        # conditioning the geometry network on per-frame learnable latent code
        if "latent_code" in input:
            network_condition = input["latent_code"]
        else:
            network_condition = None
        print('  cam_pose shape',cam_pose.shape)
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, cam_pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        idx = torch.arange(batch_size).cuda().unsqueeze(1)
        idx = idx.expand(-1, num_pixels)


        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        if self.ghostbone:
            # identity transformation for body
            transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)
        # self.geometry_network.eval()
        self.sdf.eval()
        self.deformer_network.eval()
        start_time = time.time()

        with torch.no_grad():
            print('---------------------ray_tracer')
            sdf_function = lambda x, idx: self.query_sdf(pnts_p=x,
                                                    idx=idx,
                                                    network_condition=network_condition,
                                                    pose_feature=pose_feature,
                                                    betas=expression,
                                                    transformations=transformations,
                                                    )[0]
            points, network_object_mask, dists = self.ray_tracer(sdf=sdf_function,
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs,
                                                                 idx=idx)
        end_time = time.time()
        print("Time taken for ray_tracer: ", end_time - start_time)

        # self.geometry_network.train()
        self.sdf.train()
        self.deformer_network.train()

        points = (cam_loc.unsqueeze(1) + dists.detach().reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)
        # filename = '/media/eason/edge/NeRF/IMavatar/data/experiments/yufeng/IMavatar/MVI_1810/eval/MVI_1810/rendering/points_afterrayt.ply'
        # plot.save_pcl_to_ply(filename, points.reshape(batch_size, -1, 3)[0])
        start_time = time.time()

        _, canonical_points, _, others = self.query_sdf(pnts_p=points,
                                                    idx=idx.reshape(-1),
                                                    network_condition=network_condition,
                                                    pose_feature=pose_feature,
                                                    betas=expression,
                                                    transformations=transformations,
                                                    )
        valid_mask = others['mask']
        canonical_points = canonical_points.detach()
        end_time = time.time()
        print("Time taken for query_sdf: ", end_time - start_time)
        # filename = '/media/eason/edge/NeRF/IMavatar/data/experiments/yufeng/IMavatar/MVI_1810/eval/MVI_1810/rendering/canonical_points.ply'
        # plot.save_pcl_to_ply(filename, canonical_points.reshape(batch_size, -1, 3)[0])

        # sdf_output = self.geometry_network(self.get_differentiable_non_surface(canonical_points, points, idx.reshape(-1),
        #                                                              pose_feature=pose_feature, betas=expression,
        #                                                              transformations=transformations), network_condition)[:, :1]
        sdf_output = self.grid_sampler(self.get_differentiable_non_surface(canonical_points, points, idx.reshape(-1),
                                                                     pose_feature=pose_feature, betas=expression,
                                                                     transformations=transformations), self.sdf.grid).squeeze(-1)
        sdf_output[~valid_mask] = 1
        points = points.detach()

        surface_mask = network_object_mask & object_mask if self.training else network_object_mask
        start_time = time.time()

        shapedirs, posedirs, lbs_weight = self.deformer_network.query_weights(canonical_points, mask=surface_mask)
        ray_dirs = ray_dirs.reshape(-1, 3)
        end_time = time.time()
        print("Time taken for deformer_network.query_weights: ", end_time - start_time)

        if self.training:
            start_time = time.time()
            surface_mask = network_object_mask & object_mask
            surface_canonical_points = canonical_points[surface_mask]
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]

            differentiable_surface_points = self.get_differentiable_x(pnts_c=surface_canonical_points,
                                                                      idx=idx.reshape(-1)[surface_mask],
                                                                          network_condition=network_condition,
                                                                          pose_feature=pose_feature,
                                                                          betas=expression,
                                                                          transformations=transformations,
                                                                          view_dirs=surface_ray_dirs,
                                                                          cam_loc=surface_cam_loc)
            end_time = time.time()
            print("Time taken for get_differentiable_x: ", end_time - start_time)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = canonical_points[surface_mask]


        rgb_values = torch.ones_like(points).float().cuda()
        normal_values = torch.ones_like(points).float().cuda()
        start_time = time.time()

        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask], others = self.get_rbg_value(differentiable_surface_points, idx.reshape(-1)[surface_mask], network_condition, pose_feature, expression, transformations, is_training=self.training,
                                                                  jaw_pose=torch.cat([expression, flame_pose[:, 6:9]], dim=1))
            normal_values[surface_mask] = others['normals']
        end_time = time.time()
        print("Time taken for get_rbg_value: ", end_time - start_time)

        flame_distance_values = torch.zeros(points.shape[0]).float().cuda()
        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        start_time = time.time()

        flame_distance, index_batch, _ = ops.knn_points(differentiable_surface_points.unsqueeze(0), knn_v, K=1, return_nn=True)

        end_time = time.time()
        print("Time taken for knn_points: ", end_time - start_time)

        index_batch = index_batch.reshape(-1)
        index_batch_values = torch.ones(points.shape[0]).long().cuda()
        index_batch_values[surface_mask] = index_batch
        flame_distance_values[surface_mask] = flame_distance.squeeze(0).squeeze(-1)

        output = {
            'points': points, # not differentiable
            'rgb_values': rgb_values,
            'normal_values': normal_values,
            'sdf_output': sdf_output,
            # 'sdf_values': sdf_values,

            'valid_mask': valid_mask,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'expression': expression,
            'flame_pose': flame_pose,
            'cam_pose': cam_pose,
            'index_batch': index_batch_values,
            'flame_distance': flame_distance_values,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs
        }

        if lbs_weight is not None:
            skinning_values = torch.ones(points.shape[0], 6 if self.ghostbone else 5).float().cuda()
            skinning_values[surface_mask] = lbs_weight
            output['lbs_weight'] = skinning_values
        if posedirs is not None:
            posedirs_values = torch.ones(points.shape[0], 36, 3).float().cuda()
            posedirs_values[surface_mask] = posedirs
            output['posedirs'] = posedirs_values
        if shapedirs is not None:
            shapedirs_values = torch.ones(points.shape[0], 3, 50).float().cuda()
            shapedirs_values[surface_mask] = shapedirs
            output['shapedirs'] = shapedirs_values

        if not return_sdf:
            return output
        else:
            return output, sdf_function

    def get_rbg_value(self, points, idx, network_condition, pose_feature, betas, transformations, jaw_pose=None, is_training=True):
        pnts_c = points
        others = {}
        if network_condition is not None:
            network_condition = network_condition[idx]
        pose_feature = pose_feature[idx]
        betas = betas[idx]
        transformations = transformations[idx]
        jaw_pose = jaw_pose[idx]
        _, gradients, feature_vectors,sdf = self.forward_gradient(pnts_c, network_condition, pose_feature, betas, transformations, create_graph=is_training, retain_graph=is_training)

        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)
        rgb_vals = self.rendering_network(pnts_c, normals, feature_vectors, jaw_pose=jaw_pose)

        others['normals'] = normals
        others['sdf'] = sdf

        return rgb_vals, others

    def forward_gradient(self, pnts_c, network_condition, pose_feature, betas, transformations, create_graph=True, retain_graph=True):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)
        pnts_d = self.deformer_network.forward_lbs(pnts_c, pose_feature, betas, transformations)
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            # d_out = d_out.double()*scale
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=True,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = torch.inverse(grads)
        # output = self.geometry_network(pnts_c, network_condition)
        # sdf = output[:, :1]
        # feature = output[:, 1:]
        sdf = self.grid_sampler(pnts_c, self.sdf.grid).squeeze(-1)
        feature = self.grid_sampler(pnts_c, self.color_grid.grid)
        sdf_grid = self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid
        gradients = self.neus_sdf_gradient(sdf=sdf_grid)#[1, 3, 128, 128, 128]
        grad_thetas = self.grid_sampler(pnts_c, gradients)

        # gradients = self.geometry_network.gradient(pnts_c, network_condition)
        # d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        # gradients = torch.autograd.grad(
        #     outputs=sdf,
        #     inputs=pnts_c,
        #     grad_outputs=d_output,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True)[0]
        return grads.reshape(grads.shape[0], -1), torch.nn.functional.normalize(
            torch.einsum('bi,bij->bj', grad_thetas, grads_inv), dim=1), feature,sdf
        # return sdf,  gradients, feature


    def get_differentiable_x(self, pnts_c, idx, network_condition, pose_feature, betas, transformations, view_dirs, cam_loc):
        # canonical_x : num_points, 3
        # cam_loc: num_points, 3
        # view_dirs: num_points, 3
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c = pnts_c.detach()
        pnts_c.requires_grad_(True)
        if network_condition is not None:
            network_condition = network_condition[idx]
        pose_feature = pose_feature[idx]
        betas = betas[idx]
        transformations = transformations[idx]
        deformed_x = self.deformer_network.forward_lbs(pnts_c, pose_feature, betas, transformations)

        # output = self.geometry_network(pnts_c, network_condition)
        # sdf = output[:, 0:1]
        sdf = self.grid_sampler(pnts_c, self.sdf.grid).unsqueeze(-1)
        dirs = deformed_x - cam_loc
        cross_product = torch.cross(view_dirs, dirs)
        sdf_grid = self.smooth_conv(self.sdf.grid) if self.smooth_sdf else self.sdf.grid
        gradients = self.neus_sdf_gradient(sdf=sdf_grid)#[1, 3, 128, 128, 128]
        grad_thetas = self.grid_sampler(pnts_c, gradients).unsqueeze(1)
        # print('sdf grads: ',grad_thetas)

        constant = cross_product[:,0:2]
        num_dim = constant.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(constant, requires_grad=False, device=constant.device)
            d_out[:, i] = 1
            grad = torch.autograd.grad(
                outputs=constant,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads = torch.cat((grads, grad_thetas), dim=1)
        torch.set_printoptions(profile="full")
        # print('grads: ',grads)
        identity = torch.eye(grads.size(-1), device=grads.device)
        identity_batch = identity.unsqueeze(0).repeat(grads.size(0), 1, 1)
        grads = grads + 1e-5 * identity_batch  # Add a small value multiplied by I to A
        grads_inv = torch.inverse(grads)
        constant = torch.cat([cross_product[:, 0:2], sdf], dim=1)

        differentiable_x = pnts_c.detach() - torch.einsum('bij,bj->bi', grads_inv, constant - constant.detach())
        return differentiable_x

    def get_differentiable_non_surface(self, pnts_c, points, idx, pose_feature, betas, transformations):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()

        pnts_c = pnts_c.detach()
        pnts_c.requires_grad_(True)
        pose_feature = pose_feature[idx]
        betas = betas[idx]
        transformations = transformations[idx]
        deformed_x = self.deformer_network.forward_lbs(pnts_c, pose_feature, betas, transformations)
        # points is differentiable wrt cam_loc and ray_dirs
        constant = deformed_x - points
        # filename = '/media/eason/edge/NeRF/IMavatar/data/experiments/yufeng/IMavatar/MVI_1810/eval/MVI_1810/rendering/constant.ply'
        # plot.save_pcl_to_ply(filename, constant.reshape(batch_size, -1, 3)[0])
        # filename1 = '/media/eason/edge/NeRF/IMavatar/data/experiments/yufeng/IMavatar/MVI_1810/eval/MVI_1810/rendering/deformed_x.ply'
        # plot.save_pcl_to_ply(filename1, deformed_x.reshape(batch_size, -1, 3)[0])
        # filename2 = '/media/eason/edge/NeRF/IMavatar/data/experiments/yufeng/IMavatar/MVI_1810/eval/MVI_1810/rendering/points.ply'
        # plot.save_pcl_to_ply(filename2, points.reshape(batch_size, -1, 3)[0])

        # constant: num_points, 3
        num_dim = constant.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(constant, requires_grad=False, device=constant.device)
            d_out[:, i] = 1
            grad = torch.autograd.grad(
                outputs=constant,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        differentiable_x = pnts_c.detach() - torch.einsum('bij,bj->bi', grads_inv, constant - constant.detach())
        return differentiable_x


