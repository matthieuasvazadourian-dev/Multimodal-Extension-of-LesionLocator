import os

import torch
from torch import nn
from torch.nn import functional as F
from icon_registration.mermaidlite import compute_warped_image_multiNC

class TrackNet(nn.Module):
    def __init__(self, reg_net, reg_net_patch_size, unet, unet_patch_size):
        super(TrackNet, self).__init__()
        assert len(reg_net_patch_size) == 3, "Input shape must be 3D"

        self.reg_net = reg_net
        self.unet = unet
        self.input_shape = reg_net_patch_size # [175, 175, 175]
        print(f"TrackNet input shape: {self.input_shape}", flush=True)
        print(f"TrackNet UNet patch size: {unet_patch_size}", flush=True)
        #  register as buffer
        self.register_buffer('unet_patch_size', torch.tensor(unet_patch_size))

    @staticmethod
    def _get_visualization_output_dir() -> str:
        # Use a local debug folder by default so visualize=True does not depend
        # on a machine-specific absolute path.
        output_dir = os.environ.get('LESIONLOCATOR_TRACKNET_VIS_DIR')
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), 'tracknet_visualizations')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir


    def optional_z_translation(self, x0, x1, difference_threshold=20):
        """
        Adjusts the z-dimension of two 3D tensors to align them based on their size difference.
        This function compares the z-dimensions of two input tensors `x0` and `x1`. If the difference
        in their z-dimensions exceeds the specified `difference_threshold`, it slides the smaller tensor
        along the z-dimension of the larger tensor to find the best alignment based on the lowest mean
        squared error (MSE). The tensors are then cropped to match their z-dimensions.
        Args:
            x0 (torch.Tensor): The first input tensor with shape (batch_size, channels, depth, height, width).
            x1 (torch.Tensor): The second input tensor with shape (batch_size, channels, depth, height, width).
            difference_threshold (int, optional): The maximum allowable difference in z-dimensions before
                alignment is performed. Defaults to 20.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
                - The adjusted tensor `x0` with its z-dimension cropped if necessary.
                - The adjusted tensor `x1` with its z-dimension cropped if necessary.
                - A list `[start, end]` indicating the cropping window for `x0` along the z-dimension.
                - A list `[start, end]` indicating the cropping window for `x1` along the z-dimension.
        """

        # If z dimension of images is very different, slide the smaller image to best crop the larger one
        
        x0_z, x1_z = x0.size(2), x1.size(2)
        x0_window, x1_window = [0, x0_z], [0, x1_z]
        if abs(x0_z - x1_z) > difference_threshold:
            # resample xy dimensions to 175x175
            x0_res = F.interpolate(x0, size=(x0_z, *self.input_shape[1:]), mode='trilinear', align_corners=False)
            x1_res = F.interpolate(x1, size=(x1_z, *self.input_shape[1:]), mode='trilinear', align_corners=False)
            if x0_z < x1_z:
                # slide the smaller image over the larger one and pick the spot with the lowest mse
                mse = torch.inf
                for i in range(x1_z - x0_z):
                    current_mse = F.mse_loss(x0_res, x1_res[:, :, i:i+x0_z])
                    if current_mse < mse:
                        mse = current_mse
                        x1_window = [i, i+x0_z]
            else:
                mse = torch.inf
                for i in range(x0_z - x1_z):
                    current_mse = F.mse_loss(x0_res[:, :, i:i+x1_z], x1_res)
                    if current_mse < mse:
                        mse = current_mse
                        x0_window = [i, i+x1_z]
            x0 = x0[:, :, x0_window[0]:x0_window[1]]
            x1 = x1[:, :, x1_window[0]:x1_window[1]]
        return x0, x1, x0_window, x1_window


    def get_patch_around_mask(self, prompt: torch.Tensor, is_inference: bool = False,
                              patch_size: torch.Tensor = None):
        """
        Extracts a patch around a mask from a 5D tensor input.
        This method processes a batch of 5D tensors, where each tensor represents
        a volumetric mask. For each mask in the batch, it identifies a region of
        interest (ROI) and extracts a patch centered around a foreground pixel
        or a random pixel if no foreground is present.
        Args:
            prompt (torch.Tensor): A 5D tensor of shape (batch_size, channels, depth, height, width)
                representing the input masks. The method assumes the mask is located in the first channel.
            is_inference (bool, optional): Flag indicating whether the method is being used for inference.
            patch_size (torch.Tensor, optional): Override for the patch size. If None,
                falls back to `self.unet_patch_size` (the registered buffer).
        Returns:
            list: A list of slicers for each mask in the batch. Each slicer is a list of six integers
            [xmin, xmax, ymin, ymax, zmin, zmax] defining the bounds of the patch in 3D space.
        Raises:
            AssertionError: If the input tensor `prompt` is not 5D.
        """

        assert prompt.dim() == 5, "Mask must be 5D"

        if patch_size is None:
            patch_size = self.unet_patch_size

        # Iterate over batch
        slicers = []
        for i in range(prompt.size(0)):
            mask = prompt[i, 0]

            # sample random foreground pixel
            fg_indices = torch.nonzero(mask)
            if fg_indices.size(0) == 0:
                # if there is no foreground pixel, sample a random pixel
                shape = mask.size()
                fg_idx = torch.tensor([torch.randint(0, shape[0], (1,)), torch.randint(0, shape[1], (1,)), torch.randint(0, shape[2], (1,))], device=mask.device)
            else:
                if is_inference:
                    # Get center of mass
                    fg_idx = fg_indices.float().mean(0).round().int()
                else:
                    fg_idx = fg_indices[torch.randint(0, fg_indices.size(0), (1,))][0]

            # get patch around the foreground pixel
            if fg_idx.device != patch_size.device:
                min = fg_idx - patch_size.to(fg_idx.device) // 2
                max = fg_idx + patch_size.to(fg_idx.device) // 2
            else:
                min = fg_idx - patch_size // 2
                max = fg_idx + patch_size // 2

            for i in range(3):
                if min[i] < 0:
                    if patch_size.device != fg_idx.device:
                        max[i] = patch_size.to(fg_idx.device)[i]
                    else:
                        max[i] = patch_size[i]
                    min[i] = 0
                if max[i] > mask.size(i):
                    if patch_size.device != fg_idx.device:
                        min[i] = mask.size(i) - patch_size.to(fg_idx.device)[i]
                    else:
                        min[i] = mask.size(i) - patch_size[i]
                    max[i] = mask.size(i)
                if min[i] < 0:
                    min[i] = 0
            slicers.append([min[0], max[0], min[1], max[1], min[2], max[2]])  #xmin, xmax, ymin, ymax, zmin, zmax
        return slicers
    

    # def pad_input(self, bb_input: torch.Tensor):
    #     """
    #     Pads the input tensor to match the UNet patch size.
    #     This method pads the input tensor to match the expected patch size of the UNet.
    #     Args:
    #         bb_input (torch.Tensor): The input tensor to be padded.
    #     Returns:
    #         torch.Tensor: The padded input tensor.
    #         bool: A flag indicating whether padding was applied.
    #         Tuple[int, int, int, int, int, int]: A tuple of padding values for each dimension.
    #     """
    #     if bb_input[0,0].shape != torch.Size(self.unet_patch_size):
    #         pad_x_min = (self.unet_patch_size[0] - bb_input[0,0].shape[0]) // 2
    #         pad_x_max = self.unet_patch_size[0] - bb_input[0,0].shape[0] - pad_x_min
    #         pad_y_min = (self.unet_patch_size[1] - bb_input[0,0].shape[1]) // 2
    #         pad_y_max = self.unet_patch_size[1] - bb_input[0,0].shape[1] - pad_y_min
    #         pad_z_min = (self.unet_patch_size[2] - bb_input[0,0].shape[2]) // 2
    #         pad_z_max = self.unet_patch_size[2] - bb_input[0,0].shape[2] - pad_z_min
    #         bb_input = F.pad(bb_input, (0, 0, 0, 0, pad_x_min, pad_x_max, pad_y_min, pad_y_max, pad_z_min, pad_z_max), mode='constant', value=0)
    #         return bb_input, True, (pad_x_min, pad_x_max, pad_y_min, pad_y_max, pad_z_min, pad_z_max)
    #     return bb_input, False, None
    

    # def undo_pad(self, output: torch.Tensor, pad: bool, pad_values: tuple):
    #     """
    #     Reverts padding applied to the output tensor.
    #     This method reverts padding that was previously applied to the output tensor.
    #     Args:
    #         output (torch.Tensor): The output tensor to be reverted.
    #         pad (bool): A flag indicating whether padding was applied.
    #         pad_values (tuple): A tuple of padding values for each dimension.
    #     Returns:
    #         torch.Tensor: The reverted output tensor.
    #     """
    #     if pad:
    #         pad_x_min, pad_x_max, pad_y_min, pad_y_max, pad_z_min, pad_z_max = pad_values
    #         output = F.pad(output, (0, 0, 0, 0, -pad_x_min, -pad_x_max, -pad_y_min, -pad_y_max, -pad_z_min, -pad_z_max), mode='constant', value=0)
    #     return output

    def pad_input(self, bb_input: torch.Tensor):
        """
        Pads the input tensor to match the UNet patch size.
        """
        if bb_input[0,0].shape != torch.Size(self.unet_patch_size):
            pad_z_min = (self.unet_patch_size[2] - bb_input[0,0].shape[2]) // 2
            pad_z_max = self.unet_patch_size[2] - bb_input[0,0].shape[2] - pad_z_min
            pad_y_min = (self.unet_patch_size[1] - bb_input[0,0].shape[1]) // 2
            pad_y_max = self.unet_patch_size[1] - bb_input[0,0].shape[1] - pad_y_min
            pad_x_min = (self.unet_patch_size[0] - bb_input[0,0].shape[0]) // 2
            pad_x_max = self.unet_patch_size[0] - bb_input[0,0].shape[0] - pad_x_min
            
            # Padding order for 5D: (W_left, W_right, H_left, H_right, D_left, D_right)
            # Since bb_input shape is [B, C, D, H, W], we pad: width, height, depth
            bb_input = F.pad(bb_input, (pad_z_min, pad_z_max, pad_y_min, pad_y_max, pad_x_min, pad_x_max), mode='constant', value=0)
            return bb_input, True, (pad_x_min, pad_x_max, pad_y_min, pad_y_max, pad_z_min, pad_z_max)
        return bb_input, False, None


    def undo_pad(self, output: torch.Tensor, pad: bool, pad_values: tuple):
        """
        Reverts padding applied to the output tensor.
        """
        if pad:
            pad_x_min, pad_x_max, pad_y_min, pad_y_max, pad_z_min, pad_z_max = pad_values
            
            # Calculate end indices, handling zero padding
            x_end = output.shape[2] - pad_x_max if pad_x_max > 0 else None
            y_end = output.shape[3] - pad_y_max if pad_y_max > 0 else None
            z_end = output.shape[4] - pad_z_max if pad_z_max > 0 else None
            
            # Crop the output - output shape is [B, C, D, H, W]
            output = output[:, :, 
                          pad_x_min:x_end,  # D dimension
                          pad_y_min:y_end,  # H dimension
                          pad_z_min:z_end]  # W dimension
        return output
    

    def forward(self, x0, x1, prompt, is_inference=False, x1_mask=None, visualize=False, lesion_focused=False):
        """
        Forward pass of the model.
        This method performs the forward computation for the model, including 
        resampling, registration, warping, patch extraction, and UNet inference. 
        It supports both training and inference modes.
        Args:
            x0 (torch.Tensor): The first input tensor, typically the reference image.
            x1 (torch.Tensor): The second input tensor, typically the moving image.
            prompt (torch.Tensor): A tensor representing the segmentation or prompt 
                associated with the reference image.
            is_inference (bool, optional): Flag indicating whether the method is 
                being used for inference. Defaults to False.
            x1_mask (torch.Tensor, optional): Ground truth mask for x1, only needed during training.
        Returns:
            torch.Tensor: 
                - During inference, returns the logits output tensor with the same 
                  spatial dimensions as `x1`.
                - During training, returns a tuple containing:
                    - The output tensor from the UNet.
                    - The registration loss.
                    - The cropped mask tensor for the moving image.
        """
        out_shape = x1.size()[2:]

        cuda_device = x0.device if x0.device.type == 'cuda' else None

        # Optional z-translation for inference
        if is_inference:
            x0, x1, x0_window, x1_window = self.optional_z_translation(x0, x1)
            prompt = prompt[:, :, x0_window[0]:x0_window[1]]

        # Resample to input shape on CPU
        x0 = x0.cpu()
        x1 = x1.cpu()
        prompt = prompt.cpu()

        # if batch size >1:
        if x0.size(0) > 1:
            x0_resampled = torch.stack([F.interpolate(x0[b:b+1], size=self.input_shape, mode='trilinear', align_corners=False)[0] for b in range(x0.size(0))])
            x1_resampled = torch.stack([F.interpolate(x1[b:b+1], size=self.input_shape, mode='trilinear', align_corners=False)[0] for b in range(x1.size(0))])
            prompt_resampled = torch.stack([F.interpolate(prompt[b:b+1].unsqueeze(1), size=self.input_shape, mode='nearest-exact')[0] for b in range(prompt.size(0))])
        else:
            x0_resampled = F.interpolate(x0, size=self.input_shape, mode='trilinear', align_corners=False)
            x1_resampled = F.interpolate(x1, size=self.input_shape, mode='trilinear', align_corners=False)
            prompt_resampled = F.interpolate(prompt, size=self.input_shape, mode='nearest-exact')
        
        # put back to original device
        x0_resampled = x0_resampled.to(cuda_device)
        x1_resampled = x1_resampled.to(cuda_device)
        prompt_resampled = prompt_resampled.to(cuda_device)
        
        # Registration
        # UniGradICON was pretrained on single-channel inputs (reg_input_shape has
        # channel=1). For multi-modality inputs (e.g. PET+CT early fusion) pass only
        # the first (CT) channel to the registration network; otherwise the conv
        # at the reg-net stem raises a channel-count size mismatch.
        if x0_resampled.size(1) > 1:
            reg_loss = self.reg_net(x0_resampled[:, :1], x1_resampled[:, :1])
        else:
            reg_loss = self.reg_net(x0_resampled, x1_resampled)

        time = int(torch.rand(1).item() * 1e6)
        
        if visualize:
            vis_dir = self._get_visualization_output_dir()
            import matplotlib.pyplot as plt
            fig, axis = plt.subplots(1, 4, figsize=(12, 4))
            im0 = axis[0].imshow(x0_resampled[0, 0, x0_resampled.shape[2] // 2].cpu(), cmap='gray')
            fig.colorbar(im0, ax=axis[0])
            #plt.colorbar(axis[0].imshow(x0_resampled[0, 0, x0_resampled.shape[2] // 2].cpu(), cmap='gray'), ax=axis[0])
            axis[0].set_title('x0_resampled')
            im1 = axis[1].imshow(x1_resampled[0, 0, x1_resampled.shape[2] // 2].cpu(), cmap='gray')
            fig.colorbar(im1, ax=axis[1])
            axis[1].set_title('x1_resampled')
            axis[2].imshow(x0_resampled[0, 0, x0_resampled.shape[2] // 2].cpu(), cmap='gray')
            axis[2].imshow(x1_resampled[0, 0, x1_resampled.shape[2] // 2].cpu(), cmap='jet', alpha=0.3)
            axis[2].set_title('prompt_resampled_axial')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'registration_input_resampled_{time}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
 
        print('Registration loss:', reg_loss.all_loss.item(), flush=True)
        print('Similarity loss:', reg_loss.similarity_loss.item(), flush=True)
        # Warp segmentation
        warped_prompt = compute_warped_image_multiNC(
            prompt_resampled,
            self.reg_net.phi_AB_vectorfield,
            self.reg_net.spacing,
            spline_order=0,
            zero_boundary=True
        )

        if visualize:
            # warp x0 to x1 space for visualization
            warped_x0 = compute_warped_image_multiNC(
                x0_resampled,
                self.reg_net.phi_AB_vectorfield,
                self.reg_net.spacing,
                spline_order=1,
                zero_boundary=True
            )
            
            fig, axis = plt.subplots(1, 3, figsize=(12, 4))
            axis[0].imshow(warped_x0[0, 0, warped_x0.shape[2] // 2].detach().cpu().numpy(), cmap='gray')
            axis[0].set_title('warped_x0')
            axis[1].imshow(x1_resampled[0, 0, x1_resampled.shape[2] // 2].cpu(), cmap='gray')
            axis[1].set_title('x1_resampled')
            axis[2].imshow(warped_x0[0, 0, warped_x0.shape[2] // 2].detach().cpu().numpy(), cmap='gray')
            axis[2].imshow(x1_resampled[0, 0, x1_resampled.shape[2] // 2].cpu(), cmap='jet', alpha=0.3)
            axis[2].set_title('overlay_axial')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'registration_input_warped_{time}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

            fig,axis = plt.subplots(1, 3, figsize=(12, 4))
            axis[0].imshow(warped_x0[0, 0, :, :, warped_x0.shape[4] // 2].detach().cpu().numpy(), cmap='gray')
            axis[0].set_title('warped_x0_sagittal')
            axis[1].imshow(x1_resampled[0, 0, :, :, x1_resampled.shape[4] // 2].cpu(), cmap='gray')
            axis[1].set_title('x1_resampled_sagittal')
            axis[2].imshow(warped_x0[0, 0, :, :, warped_x0.shape[4] // 2].detach().cpu().numpy(), cmap='gray')
            axis[2].imshow(x1_resampled[0, 0, :, :, x1_resampled.shape[4] // 2].cpu(), cmap='jet', alpha=0.3)
            axis[2].set_title('overlay_sagittal')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'registration_input_warped_sagittal_{time}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()


        # Move warped_prompt to cpu
        warped_prompt = warped_prompt.cpu()
        if warped_prompt.size(0) > 1:
            warped_prompt = torch.stack([F.interpolate(warped_prompt[b:b+1], size=x1.size()[2:], mode='nearest-exact')[0] for b in range(warped_prompt.size(0))])
        else:
            warped_prompt = F.interpolate(warped_prompt, size=x1.size()[2:], mode='nearest-exact')
        

        # visualize the warped prompt during inference on folow up image
        # overlay prompt on image x1_resampled
        ###############################################################################
        ###############################################################################
 
        if visualize:
            slice_idx = x0_resampled.shape[2] // 2
            import matplotlib.pyplot as plt
            import numpy as np
            import time
            vis_dir = self._get_visualization_output_dir()
            x0_np = x0_resampled[0,0].detach().cpu().numpy()
            x1_np = x1_resampled[0,0].detach().cpu().numpy()
            
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(x0_np[slice_idx], cmap='gray')
            plt.subplot(1,3,2)
            plt.title('x0 mid slice')
            plt.imshow(x1_np[slice_idx], cmap='gray')
            plt.title('x1 mid slice')

            plt.subplot(1,3,3)
            # For a Z-slice, show X and Y displacement components
            phi_AB_np = self.reg_net.phi_AB_vectorfield[0].detach().cpu().numpy()
            print(f"Deformation field shape: {phi_AB_np.shape}", flush=True)  # Should be [3, D, H, W]
            
            # Calculate in-plane magnitude for this slice (Y and X components only)
            magnitude = np.sqrt(phi_AB_np[1, slice_idx]**2 +  # Y displacement
                              phi_AB_np[2, slice_idx]**2)    # X displacement
            
            # Show the moving image (x1) as background since that's what we're registering TO
            plt.imshow(x0_np[slice_idx], cmap='gray')
            
            # Overlay magnitude with better colormap
            # Use 'hot' or 'plasma' which shows low values as dark
            im = plt.imshow(magnitude, cmap='hot', alpha=0.6, vmin=0, vmax=np.percentile(magnitude, 95))
            plt.colorbar(im, label='Total displacement (voxels)', fraction=0.046)
            
            # Downsample for quiver visualization
            step = 8  # Show fewer vectors for clarity
            # For visualizing in the (H, W) slice, use Y and X displacement components
            U = phi_AB_np[2, slice_idx, ::step, ::step]  # X displacement (W dimension)
            V = phi_AB_np[1, slice_idx, ::step, ::step]  # Y displacement (H dimension)
            
            # Create meshgrid that matches the DOWNSAMPLED arrays
            # phi_AB_np shape: [3, D, H, W] where D=264, H=384, W=384
            # After slicing at slice_idx: [H, W] = [384, 384]
            # After downsampling ::step: [H/step, W/step]
            Y_grid, X_grid = np.meshgrid(
                np.arange(0, phi_AB_np.shape[2], step),  # H dimension (384)
                np.arange(0, phi_AB_np.shape[3], step),  # W dimension (384)
                indexing='ij'
            )
            
            # Ensure shapes match by truncating if necessary
            min_y = min(Y_grid.shape[0], U.shape[0])
            min_x = min(Y_grid.shape[1], U.shape[1])
            Y_grid = Y_grid[:min_y, :min_x]
            X_grid = X_grid[:min_y, :min_x]
            U = U[:min_y, :min_x]
            V = V[:min_y, :min_x]
            
            # Overlay with quiver plot showing displacement vectors
            # Negate both to show inverse transformation (X1→X0 direction for intuitive visualization)
            plt.quiver(X_grid, Y_grid, U, -V, 
                      color='cyan', alpha=0.8, width=0.002, scale=20)
            
            plt.title(f'Deformation: X0→X1 (slice {slice_idx})\nMean: {magnitude.mean():.2f}, Max: {magnitude.max():.2f} voxels')
            plt.xlabel('X (voxels)')
            plt.ylabel('Y (voxels)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'registration_vis_slice_{slice_idx}_{int(time.time())}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()


            prompt_np = prompt.detach().cpu().numpy()[0]
            x1_mask_np = x1_mask.detach().cpu().numpy()[0] if x1_mask is not None else None
            
            if x1_mask_np is not None and np.sum(prompt_np > 0) > 0 and np.sum(x1_mask_np > 0) > 0:
                from scipy import ndimage
                prompt_com = ndimage.center_of_mass(prompt_np > 0)
                target_com = ndimage.center_of_mass(x1_mask_np > 0)
                
                # Expected displacement (in voxels)
                expected_disp = np.array(target_com) - np.array(prompt_com)[1:]
                
                # Actual deformation field at prompt center
                # phi_AB_np
                # phi_np = phi_AB_vectorfield.detach().cpu().numpy()[0]  # [3, D, H, W]
                center_idx = tuple([int(c) for c in prompt_com])
                actual_disp = phi_AB_np[:, center_idx[0], center_idx[1], center_idx[2]]
                
                print(f"\n=== DEFORMATION DIRECTION CHECK ===")
                print(f"Baseline COM: {prompt_com}")
                print(f"Target COM: {target_com}")
                print(f"Expected displacement (voxels): {expected_disp}")
                print(f"Actual deformation field: {actual_disp}")
                print(f"Dot product: {np.dot(expected_disp, actual_disp):.4f}")
                if np.dot(expected_disp, actual_disp) < 0:
                    print("⚠️  WARNING: Deformation points OPPOSITE direction!")
                print("===================================\n")


            print(f"\n=== WARPING QUALITY CHECK ===")
            print(f"Registration spacing: {self.reg_net.spacing}")
            print(f"Input shape: {self.input_shape}")
            print(f"Original segmentation:")
            print(f"  Volume: {np.sum(prompt.detach().cpu().numpy() > 0)} voxels")
            print(f"Warped segmentation:")
            print(f"  Volume: {np.sum(warped_prompt.detach().cpu().numpy() > 0)} voxels")
            print(f"  Volume preservation: {np.sum(warped_prompt.detach().cpu().numpy() > 0) / max(1, np.sum(prompt.detach().cpu().numpy() > 0)) * 100:.1f}%")

            # if visualize:
            #     import matplotlib.pyplot as plt
            #     import numpy as np
            warped_np = warped_prompt[0,0].detach().cpu().numpy()
            non_empty_mask_indices = np.nonzero(np.sum(warped_np, axis=(1,2)))
            mid_z = non_empty_mask_indices[0][len(non_empty_mask_indices[0]) // 2] if len(non_empty_mask_indices[0]) > 0 else warped_np.shape[0] // 2
            
            # Determine if we need 2 or 3 subplots based on x1_mask availability
            num_subplots = 3 if x1_mask is not None else 2
            plt.figure(figsize=(5*num_subplots, 5))

            reg_loss_val = reg_loss.all_loss if hasattr(reg_loss, 'all_loss') else reg_loss
            reg_loss_scalar = reg_loss_val.item() if hasattr(reg_loss_val, 'item') else float(reg_loss_val)

            plt.subplot(1, num_subplots, 1)
            plt.imshow(x1[0,0].detach().cpu().numpy()[mid_z], cmap='gray')
            plt.imshow(warped_np[mid_z], cmap='Greens', alpha=0.5)
            plt.title(f'Warped Prompt {mid_z} Slice, reg_loss: {reg_loss_scalar:.4f}')

            if x1_mask is not None:
                # ensure x1_mask is the same shape as warped_prompt
                plt.subplot(1, num_subplots, 2)
                assert x1_mask[0].shape == warped_prompt[0,0].shape, f"x1_mask shape {x1_mask[0].shape} does not match warped_prompt shape {warped_prompt[0,0].shape}"
                nonzero_indices = torch.sum(warped_prompt, axis = (3,4)).nonzero(as_tuple=False)[:, 2]
                nonzero_indices_target = torch.sum(x1_mask, axis = (2,3)).nonzero(as_tuple=False)[:, 1]

                indices = set(nonzero_indices.detach().cpu().numpy().tolist()).intersection(set(nonzero_indices_target.detach().cpu().numpy().tolist()))
                indices = sorted(list(indices))
                mid_idx = indices[len(indices) // 2] if len(indices) > 0 else warped_np.shape[0] // 2
                plt.imshow(x1[0,0].detach().cpu().numpy()[mid_idx], cmap='gray')
                plt.imshow(x1_mask[0].detach().cpu().numpy()[mid_idx], cmap='Reds', alpha=0.5)
                plt.imshow(warped_np[mid_idx], cmap='Greens', alpha=0.5)
                plt.title(f'Warped Prompt {mid_idx} Slice, overlap slices: {len(indices)}, reg_loss: {reg_loss_scalar:.4f}')
                plt.subplot(1, num_subplots, 3)
            else:
                plt.subplot(1, num_subplots, 2)

            # original non-warped prompt
            nonzero_indices_orig = torch.sum(prompt, axis = (3,4)).nonzero(as_tuple=False)[:, 2]
            indices_orig = set(nonzero_indices_orig.detach().cpu().numpy().tolist())
            indices_orig = sorted(list(indices_orig))
            mid_z_orig = indices_orig[len(indices_orig) // 2] if len(indices_orig) > 0 else warped_np.shape[0] // 2
            plt.imshow(x1[0,0].detach().cpu().numpy()[mid_z_orig], cmap='gray')
            plt.imshow(prompt[0,0].detach().cpu().numpy()[mid_z_orig], cmap='Reds', alpha=0.5)
            plt.title(f'Original Prompt {mid_z_orig} Slice on x1')
                        
            plt.savefig(os.path.join(vis_dir, f'warped_prompt_slice_{mid_z}_{int(time.time())}.png'))
            plt.close()

        ###############################################################################
        ###############################################################################
        
        # Patch around the warped segmentation. Use a local override for lesion-focused
        # inference; mutating `self.unet_patch_size` would replace the registered buffer
        # with a plain tensor and hard-code [64,64,64] for subsequent calls.
        if lesion_focused:
            local_patch_size = torch.tensor([64, 64, 64], device=x1.device)
        else:
            local_patch_size = None

        slicers = self.get_patch_around_mask(warped_prompt, is_inference, patch_size=local_patch_size)

        patch = torch.stack([warped_prompt[b, :, slicer[0]:slicer[1], slicer[2]:slicer[3], slicer[4]:slicer[5]] for b, slicer in enumerate(slicers)])
        
        x1 = x1.repeat(len(slicers), 1, 1, 1, 1)
        x1 = torch.stack([x1[b, :, slicer[0]:slicer[1], slicer[2]:slicer[3], slicer[4]:slicer[5]] for b, slicer in enumerate(slicers)])

        bb_input = torch.cat([x1, patch], dim=1)

        bb_input, pad, pad_properties = self.pad_input(bb_input)

        # move bb_input to original device
        bb_input = bb_input.to(cuda_device)
        
        output = self.unet(bb_input)

        # Revert padding if necessary
        output = self.undo_pad(output, pad, pad_properties)

        if is_inference: # expects batch size of 1
            assert len(slicers) == 1, "Batch size must be 1 for inference"

            slc = slicers[0]
            # adapt to x1_window
            slc[0] = slc[0] + x1_window[0]
            slc[1] = slc[1] + x1_window[0]
            
            # Immediately move output to CPU and clear all GPU tensors.
            # NOTE: x1 was moved to CPU at the top of forward(), so x1.device.type == 'cuda'
            # is always False here. Use cuda_device (captured at entry) instead.
            if cuda_device is not None:
                torch.cuda.empty_cache()

            # Move output to CPU immediately to free GPU memory
            output_cpu = output.cpu()

            # Delete GPU tensors to free memory
            del output, bb_input, patch, warped_prompt, prompt_resampled, x0_resampled, x1_resampled

            if cuda_device is not None:
                torch.cuda.empty_cache()

            # Always create on CPU to avoid any GPU memory issues
            logits_out = torch.zeros((1, 2, *out_shape), dtype=torch.float32)
            logits_out[:, 0] = 1.0  # Background class

            # Update only the patch region
            logits_out[:, :, slc[0]:slc[1], slc[2]:slc[3], slc[4]:slc[5]] = output_cpu[:, :]

            # Only try to move to GPU if the tensor is relatively small
            tensor_size_mb = logits_out.numel() * logits_out.element_size() / (1024 * 1024)

            if cuda_device is not None and tensor_size_mb < 100:  # Only move small tensors to GPU
                try:
                    logits_out = logits_out.to(cuda_device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"Keeping output tensor on CPU - GPU memory insufficient for {tensor_size_mb:.2f} MB tensor")
                    else:
                        raise e
            else:
                print("Keeping output tensor on CPU due to size or device constraints")

            return logits_out, reg_loss
        else:
            # x1_mask is only needed to be passed to the method during training
            if x1_mask is None:
                # return output, reg_loss, None
                raise ValueError("x1_mask must be provided when is_inference=False (training mode)")
            
            if len(x1_mask.size()) == 4:
                x1_mask = x1_mask.unsqueeze(1)  # add channel dimension if missing
            x1_mask = torch.stack([x1_mask[b, :, slicer[0]:slicer[1], slicer[2]:slicer[3], slicer[4]:slicer[5]] for b, slicer in enumerate(slicers)])
            return output, reg_loss, x1_mask # for training
