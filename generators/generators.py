"""Implicit generator for 3D volumes"""

import torch.nn as nn
import torch

from .volumetric_rendering import *
from .decoder import ResDecoder
from .utils import *


class ImplicitGenerator3d(nn.Module):
    def __init__(self, siren, z_dim, shading, view_condition, light_condition, surf_track, ldist=None, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.shading = shading
        self.surf_track = surf_track
        self.siren = siren(output_dim=4, z_dim=self.z_dim, input_dim=3, shading=shading,
                           view_condition=view_condition, light_condition=light_condition, device=None)
        if self.surf_track:
            self.surfacenet = ResDecoder(4608, nf=16)
        self.epoch = 0
        self.step = 0
        self.ldist = ldist if ldist is not None else LSampler(device=self.siren.device)
        self.init_cam2world()

    def set_device(self, device):
        self.device = device
        self.siren.device = device
        self.ldist.device = device
        if self.surf_track:
            self.surfacenet.device = device
        self.generate_avg_frequencies()
        self.cam2world_matrix = self.cam2world_matrix.to(device)

    def init_cam2world(self):
        camera_origin = torch.zeros((1,3))
        camera_origin[:, 2] = 1
        forward_vector = -camera_origin
        self.cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin)

    def lambertian_shading(self, inputs, l, l_ratio=1):
        normal, albedo = inputs['normal'], inputs['rgb']
        b = normal.size(0)
        rand_light_dxy = l[:,2:]
        rand_light_d = torch.cat([rand_light_dxy, torch.ones(b, 1).to(rand_light_dxy)], 1)
        rand_light_d = rand_light_d / (torch.norm(rand_light_d, dim=-1, keepdim=True) + 1e-7)
        transformed_light = torch.bmm(self.cam2world_matrix.expand(b,4,4)[..., :3, :3], rand_light_d.reshape(b,1,3).permute(0,2,1)).permute(0, 2, 1).expand_as(normal)
        rand_diffuse_shading = (normal * transformed_light).sum(-1, keepdim=True).clamp(min=0, max=1)
        rand_diffuse_shading[torch.isnan(rand_diffuse_shading)] = 1.0
        ambience = l[:,None,:1]/2+0.5
        diffuse = l[:,None,1:2]/2+0.5
        rand_shading = ambience + diffuse*rand_diffuse_shading
        # smoothly transfer from no shading to shading
        rand_shading = l_ratio * rand_shading + (1 - l_ratio)
        rgb = (albedo * rand_shading).clamp(min=0, max=1)
        inputs['albedo'] = albedo
        inputs['rgb'] = rgb
        inputs['shading'] = rand_shading
        return inputs

    def forward(self, z, l, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample,
                sample_dist=None, lock_view_dependence=False, delta=-1, pose=None, l_ratio=1, rt_normal=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """

        batch_size = z.shape[0]
        with_grad = torch.is_grad_enabled()

        # Generate initial camera rays and sample points.
        with torch.no_grad():
            if pose is None:
                camera_origin, pitch, yaw = sample_camera_positions(n=batch_size, r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=self.device, mode=sample_dist)
            else:
                pitch, yaw = pose[:,:1], pose[:,1:2]
                camera_origin = pose2origin(self.device, pitch, yaw, batch_size, 1)
            if self.surf_track:
                with torch.set_grad_enabled(with_grad):
                    freq, phase = self.siren.mapping_network(z)
                    freq, phase = freq.detach(), phase.detach()
                    pose_scaled = (torch.cat([pitch, yaw], -1) - math.pi/2) * 10
                    freq_phase = torch.cat([freq, phase], -1)/10
                    pred = self.surfacenet(freq_phase, pose_scaled)
                    depth_pred = pred[:,0,...] # 2nd channel not used for now
                    depth_pred = ray_start + depth_pred * (ray_end - ray_start)
                if delta > 0:
                    sample_depth = resize(depth_pred, img_size)
                    delta = torch.ones_like(sample_depth) * delta
                    sample_depth = torch.max(sample_depth, ray_start+delta/2)
                    sample_depth = torch.min(sample_depth, ray_end-delta/2)
                    points_cam, z_vals, rays_d_cam = get_rays_from_depth(batch_size, num_steps, sample_depth, delta, resolution=(img_size, img_size), device=self.device, fov=fov) # batch_size, pixels, num_steps, 1
                else:
                    points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            else:
                depth_pred = None
                points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins = transform_sampled_points(points_cam, z_vals, rays_d_cam, camera_origin, self.device)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Model prediction on course points
        coarse_output = self.siren(transformed_points, z, l, ray_directions=transformed_ray_directions_expanded, rt_normal=rt_normal)
        for k, v in coarse_output.items():
            coarse_output[k] = v.reshape(batch_size, img_size * img_size, num_steps, -1)

        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                course_results = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
                weights = course_results['weights']
                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # Model prediction on re-sampled find points
            fine_output = self.siren(fine_points, z, l, ray_directions=transformed_ray_directions_expanded, rt_normal=rt_normal)
            for k, v in fine_output.items():
                fine_output[k] = v.reshape(batch_size, img_size * img_size, num_steps, -1)

            # Combine course and fine points
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = {}
            for k, v in coarse_output.items():
                all_outputs[k] = torch.cat([fine_output[k], v], dim = -2)
                # Target sizes: [-1, -1, -1, 4].  Tensor sizes: [240, 512, 12]
                all_outputs[k] = torch.gather(all_outputs[k], -2, indices.expand(-1, -1, -1, all_outputs[k].size(-1)))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        # Create images with NeRF
        results = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        if self.shading:
            results = self.lambertian_shading(results, l, l_ratio=l_ratio)
        results['depth_std'] = course_results['depth_std']
        for k in ['rgb', 'rgb_refer']:
            if k in results:
                results[k] = results[k].reshape(batch_size, img_size, img_size, 3)
                results[k] = results[k].permute(0, 3, 1, 2).contiguous() * 2 - 1
        if 'normal' in results:
            results['normal'] = results['normal'].reshape(batch_size, img_size, img_size, 3).permute(0, 3, 1, 2).contiguous()
        for k in ['depth', 'depth_std']:
            results[k] = results[k].reshape(batch_size, img_size, img_size).contiguous()
            if depth_pred is not None:
                results[k] = resize(results[k], depth_pred.size(-1))
        results['depth'] = results['depth'].clamp(min=ray_start, max=ray_end)
        results['pose'] = torch.cat([pitch, yaw], -1)
        results['depth_pred'] = depth_pred

        return results


    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""

        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)


    def staged_forward(self, z, l, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=0.8, lock_view_dependence=False, max_batch_size=50000, sample_dist=None, hierarchical_sample=False, delta=-1, pose=None, l_ratio=1, rt_normal=False, **kwargs):
        batch_size = z.shape[0]

        self.generate_avg_frequencies()

        with torch.no_grad():

            raw_frequencies, raw_phase_shifts = self.siren.mapping_network(z)

            truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)

            if pose is None:
                camera_origin, pitch, yaw = sample_camera_positions(n=batch_size, r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=self.device, mode=sample_dist)
            else:
                pitch, yaw = pose[:,:1], pose[:,1:2]
                camera_origin = pose2origin(self.device, pitch, yaw, batch_size, 1)
            if self.surf_track:
                pose_scaled = (torch.cat([pitch, yaw], -1) - math.pi/2) * 10
                freq_phase = torch.cat([truncated_frequencies, truncated_phase_shifts], -1)/10
                pred = self.surfacenet(freq_phase, pose_scaled)
                depth_pred = pred[:,0,...] # 2nd channel not used for now
                depth_pred = ray_start + depth_pred * (ray_end - ray_start)
                if delta > 0:
                    sample_depth = resize(depth_pred, img_size)
                    delta = torch.ones_like(sample_depth) * delta
                    sample_depth = torch.max(sample_depth, ray_start+delta/2)
                    sample_depth = torch.min(sample_depth, ray_end-delta/2)
                    points_cam, z_vals, rays_d_cam = get_rays_from_depth(batch_size, num_steps, sample_depth, delta, resolution=(img_size, img_size), device=self.device, fov=fov) # batch_size, pixels, num_steps, 1
                else:
                    points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            else:
                depth_pred = None
                points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins = transform_sampled_points(points_cam, z_vals, rays_d_cam, camera_origin, self.device)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # BATCHED SAMPLE
            coarse_output = {}
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    output = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], l=l[b:b+1],
                                                                              ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], rt_normal=rt_normal)
                    for k, v in output.items():
                        if not k in coarse_output:
                            coarse_output[k] = torch.zeros((batch_size, transformed_points.shape[1], v.size(-1)), device=self.device)
                        coarse_output[k][b:b+1, head:tail] = v
                    head += max_batch_size

            for k, v in coarse_output.items():
                coarse_output[k] = v.reshape(batch_size, img_size * img_size, num_steps, -1)
            # END BATCHED SAMPLE


            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])['weights']
                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                     num_steps, det=False).detach().to(self.device) # batch_size, num_pixels**2, num_steps
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                # BATCHED SAMPLE
                fine_output = {}
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        output = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], l=l[b:b+1],
                                                                                  ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], rt_normal=rt_normal)
                        for k, v in output.items():
                            if not k in fine_output:
                                fine_output[k] = torch.zeros((batch_size, fine_points.shape[1], v.size(-1)), device=self.device)
                            fine_output[k][b:b+1, head:tail] = v
                        head += max_batch_size

                for k, v in fine_output.items():
                    fine_output[k] = v.reshape(batch_size, img_size * img_size, num_steps, -1)
                # END BATCHED SAMPLE

                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = {}
                for k, v in coarse_output.items():
                    all_outputs[k] = torch.cat([fine_output[k], v], dim = -2)
                    all_outputs[k] = torch.gather(all_outputs[k], -2, indices.expand(-1, -1, -1, all_outputs[k].size(-1)))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals

            results = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'])
            if self.shading:
                results = self.lambertian_shading(results, l, l_ratio=l_ratio)
            for k in ['rgb', 'rgb_refer', 'albedo']:
                if k in results:
                    results[k] = results[k].reshape(batch_size, img_size, img_size, 3)
                    results[k] = results[k].permute(0, 3, 1, 2).contiguous() * 2 - 1
            if 'normal' in results:
                results['normal'] = results['normal'].reshape(batch_size, img_size, img_size, 3).permute(0, 3, 1, 2).contiguous()
            for k in ['depth', 'shading']:
                if k in results:
                    results[k] = results[k].reshape(batch_size, img_size, img_size).contiguous()
            results['depth'] = results['depth'].clamp(min=ray_start, max=ray_end)
            results['pose'] = torch.cat([pitch, yaw], -1)
        results['depth_pred'] = depth_pred

        return results

    # Used for rendering interpolations
    def staged_forward_with_frequencies(self, z, frequencies, phase_shifts, l, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, psi=0.8, lock_view_dependence=False, max_batch_size=50000, sample_dist=None, hierarchical_sample=False, delta=-1, pose=None, l_ratio=1, rt_normal=False, **kwargs):
        batch_size = z.shape[0]

        self.generate_avg_frequencies()

        with torch.no_grad():

            truncated_frequencies = self.avg_frequencies + psi * (frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (phase_shifts - self.avg_phase_shifts)

            if pose is None:
                camera_origin, pitch, yaw = sample_camera_positions(n=batch_size, r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=self.device, mode=sample_dist)
            else:
                pitch, yaw = pose[:,:1], pose[:,1:2]
                camera_origin = pose2origin(self.device, pitch, yaw, batch_size, 1)
            if self.surf_track:
                pose_scaled = (torch.cat([pitch, yaw], -1) - math.pi/2) * 10
                freq_phase = torch.cat([truncated_frequencies, truncated_phase_shifts], -1)/10
                pred = self.surfacenet(freq_phase, pose_scaled)
                if pred.size(2) > img_size:
                    pred = F.interpolate(pred, img_size, mode='area')
                elif pred.size(2) < img_size:
                    pred = F.interpolate(pred, img_size, mode='bilinear')
                depth_pred = pred[:,0,...] # 2nd channel not used for now
                depth_pred = ray_start + depth_pred * (ray_end - ray_start)
                if delta > 0:
                    sample_depth = resize(depth_pred, img_size)
                    delta = torch.ones_like(sample_depth) * delta
                    sample_depth = torch.max(sample_depth, ray_start+delta/2)
                    sample_depth = torch.min(sample_depth, ray_end-delta/2)
                    points_cam, z_vals, rays_d_cam = get_rays_from_depth(batch_size, num_steps, sample_depth, delta, resolution=(img_size, img_size), device=self.device, fov=fov) # batch_size, pixels, num_steps, 1
                else:
                    points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            else:
                depth_pred = None
                points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins = transform_sampled_points(points_cam, z_vals, rays_d_cam, camera_origin, self.device)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # BATCHED SAMPLE
            coarse_output = {}
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    output = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], l=l[b:b+1],
                                                                              ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], rt_normal=rt_normal)
                    for k, v in output.items():
                        if not k in coarse_output:
                            coarse_output[k] = torch.zeros((batch_size, transformed_points.shape[1], v.size(-1)), device=self.device)
                        coarse_output[k][b:b+1, head:tail] = v
                    head += max_batch_size

            for k, v in coarse_output.items():
                coarse_output[k] = v.reshape(batch_size, img_size * img_size, num_steps, -1)
            # END BATCHED SAMPLE

            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])['weights']
                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                     num_steps, det=False).detach().to(self.device) # batch_size, num_pixels**2, num_steps
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                # BATCHED SAMPLE
                fine_output = {}
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        output = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], l=l[b:b+1],
                                                                                  ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail], rt_normal=rt_normal)
                        for k, v in output.items():
                            if not k in fine_output:
                                fine_output[k] = torch.zeros((batch_size, fine_points.shape[1], v.size(-1)), device=self.device)
                            fine_output[k][b:b+1, head:tail] = v
                        head += max_batch_size

                for k, v in fine_output.items():
                    fine_output[k] = v.reshape(batch_size, img_size * img_size, num_steps, -1)
                # END BATCHED SAMPLE

                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = {}
                for k, v in coarse_output.items():
                    all_outputs[k] = torch.cat([fine_output[k], v], dim = -2)
                    all_outputs[k] = torch.gather(all_outputs[k], -2, indices.expand(-1, -1, -1, all_outputs[k].size(-1)))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals

            results = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'])
            if self.shading:
                results = self.lambertian_shading(results, l, l_ratio=l_ratio)
            for k in ['rgb', 'rgb_refer', 'albedo']:
                if k in results:
                    results[k] = results[k].reshape(batch_size, img_size, img_size, 3)
                    results[k] = results[k].permute(0, 3, 1, 2).contiguous() * 2 - 1
            if 'normal' in results:
                results['normal'] = results['normal'].reshape(batch_size, img_size, img_size, 3).permute(0, 3, 1, 2).contiguous()
            for k in ['depth', 'shading']:
                if k in results:
                    results[k] = results[k].reshape(batch_size, img_size, img_size).contiguous()
            results['depth'] = results['depth'].clamp(min=ray_start, max=ray_end)
            results['pose'] = torch.cat([pitch, yaw], -1)
        results['depth_pred'] = depth_pred

        return results


    def forward_with_frequencies(self, z, frequencies, phase_shifts, l, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean,
                                 hierarchical_sample, sample_dist=None, lock_view_dependence=False, delta=-1, l_ratio=1, pose=None, rt_normal=False, **kwargs):
        batch_size = frequencies.shape[0]
        with_grad = torch.is_grad_enabled()

        with torch.no_grad():
            if pose is None:
                camera_origin, pitch, yaw = sample_camera_positions(n=batch_size, r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, horizontal_mean=h_mean, vertical_mean=v_mean, device=self.device, mode=sample_dist)
            else:
                pitch, yaw = pose[:,:1], pose[:,1:2]
                camera_origin = pose2origin(self.device, pitch, yaw, batch_size, 1)
            if self.surf_track:
                with torch.set_grad_enabled(with_grad):
                    freq, phase = self.siren.mapping_network(z)
                    freq, phase = freq.detach(), phase.detach()
                    pose_scaled = (torch.cat([pitch, yaw], -1) - math.pi/2) * 10
                    freq_phase = torch.cat([freq, phase], -1)/10
                    pred = self.surfacenet(freq_phase, pose_scaled)
                    depth_pred = pred[:,0,...] # 2nd channel not used for now
                    depth_pred = ray_start + depth_pred * (ray_end - ray_start)
                if delta > 0:
                    sample_depth = resize(depth_pred, img_size)
                    delta = torch.ones_like(sample_depth) * delta
                    sample_depth = torch.max(sample_depth, ray_start+delta/2)
                    sample_depth = torch.min(sample_depth, ray_end-delta/2)
                    points_cam, z_vals, rays_d_cam = get_rays_from_depth(batch_size, num_steps, sample_depth, delta, resolution=(img_size, img_size), device=self.device, fov=fov) # batch_size, pixels, num_steps, 1
                else:
                    points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            else:
                depth_pred = None
                points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins = transform_sampled_points(points_cam, z_vals, rays_d_cam, camera_origin, self.device)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points, frequencies, phase_shifts, l=l, ray_directions=transformed_ray_directions_expanded, rt_normal=rt_normal)
        for k, v in coarse_output.items():
            coarse_output[k] = v.reshape(batch_size, img_size * img_size, num_steps, -1)

        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])['weights']

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                #### Start new importance sampling
                # RuntimeError: Sizes of tensors must match except in dimension 1. Got 3072 and 6144 (The offending index is 0)
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach() # batch_size, num_pixels**2, num_steps
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

            fine_output = self.siren.forward_with_frequencies_phase_shifts(fine_points, frequencies, phase_shifts, l=l, ray_directions=transformed_ray_directions_expanded, rt_normal=rt_normal)
            for k, v in fine_output.items():
                fine_output[k] = v.reshape(batch_size, img_size * img_size, num_steps, -1)

            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = {}
            for k, v in coarse_output.items():
                all_outputs[k] = torch.cat([fine_output[k], v], dim = -2)
                all_outputs[k] = torch.gather(all_outputs[k], -2, indices.expand(-1, -1, -1, all_outputs[k].size(-1)))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        results = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        if self.shading:
            results = self.lambertian_shading(results, l, l_ratio=l_ratio)
        for k in ['rgb', 'rgb_refer']:
            if k in results:
                results[k] = results[k].reshape(batch_size, img_size, img_size, 3)
                results[k] = results[k].permute(0, 3, 1, 2).contiguous() * 2 - 1
        if 'normal' in results:
            results['normal'] = results['normal'].reshape(batch_size, img_size, img_size, 3).permute(0, 3, 1, 2).contiguous()
        for k in ['depth', 'shading']:
            if k in results:
                results[k] = results[k].reshape(batch_size, img_size, img_size).contiguous()
        results['depth'] = results['depth'].clamp(min=ray_start, max=ray_end)
        results['pose'] = torch.cat([pitch, yaw], -1)

        return results
