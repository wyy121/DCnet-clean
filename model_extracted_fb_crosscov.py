from collections.abc import Iterable
from math import ceil, prod
from typing import Optional

import torch
import torch.nn as nn

from utils import get_activation_class


'''class LowRankModulation(nn.Module):
    def __init__(self, in_channels: int, spatial_size: tuple[int, int], alpha: float = 1.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.alpha = alpha
        
        self.spatial_average = nn.AdaptiveAvgPool2d((1, 1))
        self.rank_one_vec_h = nn.Linear(in_channels, spatial_size[0], bias=True)
        self.rank_one_vec_w = nn.Linear(in_channels, spatial_size[1], bias=True)
        
        # 添加 sigmoid 门控
        self.gate = nn.Sigmoid()  # ← 直接添加

    def forward(self, cue: torch.Tensor, mixture: torch.Tensor):
        e = self.spatial_average(cue).flatten(1)
        
        hvec = self.rank_one_vec_h(e)
        wvec = self.rank_one_vec_w(e)
        
        rank_one_matrix = torch.bmm(hvec.unsqueeze(-1), wvec.unsqueeze(-2))
        rank_one_matrix = rank_one_matrix.unsqueeze(1)
        
        # 应用 sigmoid 门控
        spatial_gate = self.gate(rank_one_matrix)  # ← 关键修改
        gamma = e.unsqueeze(-1).unsqueeze(-1) * spatial_gate
        
        return mixture * (1 + self.alpha * gamma)
        #return mixture * self.alpha * gamma'''

class LowRankModulation(nn.Module):
    def __init__(self, in_channels, spatial_size: tuple[int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size

        self.spatial_average = nn.AdaptiveAvgPool2d((1, 1))
        self.rank_one_vec_h = nn.Linear(in_channels, spatial_size[0])
        self.rank_one_vec_w = nn.Linear(in_channels, spatial_size[1])
        self.gate = nn.Sigmoid()  # ← 添加 sigmoid 门控

    def forward(self, cue: torch.Tensor, mixture: torch.Tensor):
        x = self.spatial_average(cue)
        x = x.flatten(1)
        hvec = self.rank_one_vec_h(x)
        wvec = self.rank_one_vec_w(x)

        rank_one_matrix = torch.bmm(
            hvec.unsqueeze(-1), wvec.unsqueeze(-2)
        ).unsqueeze(-3)
        
        spatial_gate = self.gate(rank_one_matrix)  # ← 应用 sigmoid
        
        rank_one_tensor = x.unsqueeze(-1).unsqueeze(-1) * spatial_gate

        return mixture * rank_one_tensor


class LowRankPerturbation(nn.Module):
    def __init__(self, in_channels: int, spatial_size: tuple[int, int]):
        """
        Initializes the EI model.

        Args:
            in_channels (int): The number of input channels.
            spatial_size (tuple[int, int]): The spatial size of the input.

        """
        super().__init__()
        # Initialize the weight and bias matrices
        self.W = nn.Parameter(torch.randn(1, in_channels, spatial_size[0], 1))
        self.bias = nn.Parameter(
            torch.randn(1, in_channels, spatial_size[0], 1)
        )

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the model.

        Args:
            cue (torch.Tensor): The cue tensor.
            mixture (torch.Tensor): The mixture tensor.

        Returns:
            torch.Tensor: The output tensor after adding the rank one perturbation to the mixture.
        """
        # Compute the rank one matrix
        rank_one_vector = torch.matmul(input, self.W) + self.bias
        rank_one_perturbation = torch.matmul(
            rank_one_vector, rank_one_vector.transpose(-2, -1)
        )
        return rank_one_perturbation


class Conv2dPositive(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        self.weight.data = torch.relu(self.weight.data)
        if self.bias is not None:
            self.bias.data = torch.relu(self.bias.data)
        return super().forward(*args, **kwargs)


class Conv2dEIRNNCell(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        input_dim: int,
        h_pyr_dim: int,
        h_inter_dim: int,
        fb_dim: int = 0,
        exc_kernel_size: tuple[int, int] = (5, 5),
        inh_kernel_size: tuple[int, int] = (5, 5),
        immediate_inhibition: bool = False,
        exc_rectify: Optional[str] = None,
        inh_rectify: Optional[str] = "neg",
        pool_kernel_size: tuple[int, int] = (5, 5),
        pool_stride: tuple[int, int] = (2, 2),
        bias: bool = True,
        pre_inh_activation: Optional[str] = "tanh",
        post_inh_activation: Optional[str] = None,
        fb_to_inter: bool = False,
    ):
        """
        Initialize the ConvRNNEICell.

        Args:
            input_size (tuple[int, int]): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            h_pyr_dim (int, optional): Number of channels of the excitatory pyramidal tensor. Default is 4.
            h_inter_dims (tuple[int], optional): Number of channels of the interneuron tensors. Default is (4).
            fb_dim (int, optional): Number of channels of the feedback excitatory tensor. Default is 0.
            exc_kernel_size (tuple[int, int], optional): Size of the kernel for excitatory convolution. Default is (5, 5).
            inh_kernel_size (tuple[int, int], optional): Size of the kernel for inhibitory convolution. Default is (5, 5).
            num_compartments (int, optional): Number of compartments. Default is 3.
            immediate_inhibition (bool, optional): Whether to use immediate inhibition. Default is False.
            pool_kernel_size (tuple[int, int], optional): Size of the kernel for pooling. Default is (5, 5).
            pool_stride (tuple[int, int], optional): Stride for pooling. Default is (2, 2).
            bias (bool, optional): Whether to add bias. Default is True.
            activation (str, optional): Activation function to use. Only 'tanh' and 'relu' activations are supported. Default is "relu".
        """
        super().__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.h_pyr_dim = h_pyr_dim
        self.h_inter_dim = h_inter_dim
        self.fb_dim = fb_dim
        self.use_fb = fb_dim > 0
        self.fb_to_inter = fb_to_inter
        self.immediate_inhibition = immediate_inhibition
        self.pool_stride = pool_stride
        if isinstance(pre_inh_activation, (list, tuple)):
            activations = []
            for activation in pre_inh_activation:
                activations.append(get_activation_class(activation)())
            self.pre_inh_activation = nn.Sequential(*activations)
        else:
            self.pre_inh_activation = get_activation_class(
                pre_inh_activation
            )()
        if isinstance(post_inh_activation, (list, tuple)):
            activations = []
            for activation in post_inh_activation:
                activations.append(get_activation_class(activation)())
            self.post_inh_activation = nn.Sequential(*activations)
        else:
            self.post_inh_activation = get_activation_class(
                post_inh_activation
            )()
        self.output_dim = h_pyr_dim
        self.output_size = (
            ceil(input_size[0] / pool_stride[0]),
            ceil(input_size[1] / pool_stride[1]),
        )

        # One learnable time constant per channel, shared across spatial positions.
        self.tau_pyr = nn.Parameter(torch.randn((1, h_pyr_dim, 1, 1)))
        if h_inter_dim > 0:
            self.tau_inter = nn.Parameter(
                torch.randn((1, self.h_inter_dim, 1, 1))
            )
        '''# 初始化方差为 0.01，即标准差为 0.1
        init_std = 0.1
        self.tau_pyr = nn.Parameter(torch.randn((1, h_pyr_dim, 1, 1)) * init_std)

        if h_inter_dim > 0:
            self.tau_inter = nn.Parameter(
                torch.randn((1, self.h_inter_dim, 1, 1)) * init_std
            )'''
    
        '''self.tau_pyr = nn.Parameter(torch.randn((1, h_pyr_dim, 1, 1)) * 0.5)
        if h_inter_dim > 0:
            self.tau_inter = nn.Parameter(
                torch.randn((1, self.h_inter_dim, 1, 1)) * 0.5
            )'''
        

        if exc_rectify == "pos":
            Conv2dExc = Conv2dPositive
        elif exc_rectify is None:
            Conv2dExc = nn.Conv2d
        else:
            raise ValueError("pyr_rectify must be 'pos' or None.")

        if inh_rectify == "pos":
            Conv2dInh = Conv2dPositive
        elif inh_rectify is None:
            Conv2dInh = nn.Conv2d

        # Initialize excitatory convolutional layers
        self.conv_exc_pyr = Conv2dExc(
            in_channels=input_dim + h_pyr_dim + fb_dim,
            out_channels=h_pyr_dim,
            kernel_size=exc_kernel_size,
            padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
            bias=bias,
        )

        if h_inter_dim > 0:
            conv_exc_inter_in_channels = h_pyr_dim + input_dim
            if fb_to_inter:
                conv_exc_inter_in_channels += fb_dim
            self.conv_exc_inter = Conv2dExc(
                in_channels=conv_exc_inter_in_channels,
                out_channels=self.h_inter_dim,
                kernel_size=exc_kernel_size,
                stride=1,
                padding=(exc_kernel_size[0] // 2, exc_kernel_size[1] // 2),
                bias=bias,
            )

        # Initialize inhibitory convolutional layers
        if h_inter_dim > 0:
            self.conv_inh = Conv2dInh(
                in_channels=h_inter_dim,
                out_channels=h_pyr_dim,
                kernel_size=inh_kernel_size,
                padding=(inh_kernel_size[0] // 2, inh_kernel_size[1] // 2),
                bias=bias,
            )

        # Initialize output pooling layer
        self.out_pool = nn.AvgPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=(pool_kernel_size[0] // 2, pool_kernel_size[1] // 2),
        )

    def init_hidden(self, batch_size, init_mode="zeros", device=None):
        """
        Initializes the hidden state tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.
            device (torch.device, optional): The device to initialize the tensor on. Default is None.
            init_mode (str, optional): The initialization mode. Can be "zeros" or "normal". Default is "zeros".

        Returns:
            torch.Tensor: The initialized excitatory hidden state tensor.
            torch.Tensor: The initialized inhibitory hidden state tensor.
        """

        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "normal":
            func = torch.randn
        else:
            raise ValueError("Invalid init_mode. Must be 'zeros' or 'normal'.")
        return (
            func(batch_size, self.h_pyr_dim, *self.input_size, device=device),
            (
                func(
                    batch_size,
                    self.h_inter_dim,
                    *self.input_size,
                    device=device,
                )
                if self.h_inter_dim > 0
                else None
            ),
        )

    def init_fb(self, batch_size, init_mode="zeros", device=None):
        """
        Initializes the output tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.
            device (torch.device, optional): The device to initialize the tensor on. Default is None.
            init_mode (str, optional): The initialization mode. Can be "zeros" or "normal". Default is "zeros".

        Returns:
            torch.Tensor: The initialized output tensor.
        """
        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "normal":
            func = torch.randn
        else:
            raise ValueError("Invalid init_mode. Must be 'zeros' or 'normal'.")
        return func(batch_size, self.fb_dim, *self.input_size, device=device)

    def init_out(self, batch_size, init_mode="zeros", device=None):
        """
        Initializes the output tensor for the cRNN_EI model.

        Args:
            batch_size (int): The size of the input batch.
            device (torch.device, optional): The device to initialize the tensor on. Default is None.
            init_mode (str, optional): The initialization mode. Can be "zeros" or "normal". Default is "zeros".

        Returns:
            torch.Tensor: The initialized output tensor.
        """
        if init_mode == "zeros":
            func = torch.zeros
        elif init_mode == "normal":
            func = torch.randn
        else:
            raise ValueError("Invalid init_mode. Must be 'zeros' or 'normal'.")
        return func(
            batch_size, self.output_dim, *self.output_size, device=device
        )
    
    def forward(
        self,
        input: torch.Tensor,
        h_pyr: torch.Tensor,
        h_inter: torch.Tensor,
        fb: torch.Tensor = None,
    ):
        """
        Performs forward pass of the cRNN_EI model.

        Args:
            input (torch.Tensor): Input tensor of shape (b, c, h, w).
            h_pyr (torch.Tensor): Current pyramidal hidden state.
            h_inter (torch.Tensor): Current interneuronal hidden state.
            fb (torch.Tensor, optional): Feedback input.

        Returns:
            torch.Tensor: Next pyramidal hidden state.
            torch.Tensor: Next interneuronal hidden state.
            torch.Tensor: Output tensor after pooling.
        """
        if self.use_fb and fb is None:
            raise ValueError("If use_fb is True, fb_exc must be provided.")

        # Compute the excitations for pyramidal cells
        exc_cat = [input, h_pyr]
        if self.use_fb:
            exc_cat.append(fb)
        exc_pyr = self.pre_inh_activation(
            self.conv_exc_pyr(torch.cat(exc_cat, dim=1))
        )

        '''if self.h_inter_dim > 0:
            # Compute the excitations for interneurons
            exc_inter = self.pre_inh_activation(
                self.conv_exc_inter(torch.cat(exc_cat, dim=1))
            )'''
        if self.h_inter_dim > 0:
            # Inter: 根据 fb_to_inter 决定是否包含 fb
            inh_cat = [input, h_pyr]
            if self.use_fb and self.fb_to_inter:
                inh_cat.append(fb)
            
            exc_inter = self.pre_inh_activation(
                self.conv_exc_inter(torch.cat(inh_cat, dim=1))
            )

            # Compute candidate inhibitory state
            cnm_inter = self.post_inh_activation(exc_inter)
            
            # Update interneuronal state FIRST
            tau_inter = torch.sigmoid(self.tau_inter)
            h_next_inter = (1 - tau_inter) * h_inter + tau_inter * cnm_inter
            
            # THEN compute inhibition from the UPDATED interneuronal state
            inh_pyr = self.pre_inh_activation(self.conv_inh(h_next_inter))

        # Compute candidate pyramidal state
        cnm_pyr = self.post_inh_activation(exc_pyr - inh_pyr)
        
        # Update pyramidal state
        tau_pyr = torch.sigmoid(self.tau_pyr)
        h_next_pyr = (1 - tau_pyr) * h_pyr + tau_pyr * cnm_pyr

        # Pool the output
        out = self.out_pool(h_next_pyr)

        return h_next_pyr, h_next_inter, out
    
    '''def forward(
        self,
        input: torch.Tensor,
        h_pyr: torch.Tensor,
        h_inter: torch.Tensor,
        fb: torch.Tensor = None,
    ):
        """
        Performs forward pass of the cRNN_EI model.

        Args:
            input (torch.Tensor): Input tensor of shape (b, c, h, w).
                The input is actually the target_model.
            h (torch.Tensor): Current hidden and cell states respectively
                of shape (b, c_hidden, h, w).

        Returns:
            torch.Tensor: Next hidden state of shape (b, c_hidden*2, h, w).
            torch.Tensor: Output tensor after pooling of shape (b, c_hidden*2, h', w').
        """
        if self.use_fb and fb is None:
            raise ValueError("If use_fb is True, fb_exc must be provided.")

        # Compute the excitations for pyramidal cells
        exc_cat = [input, h_pyr]
        if self.use_fb:
            exc_cat.append(fb)
        exc_pyr = self.pre_inh_activation(
            self.conv_exc_pyr(torch.cat(exc_cat, dim=1))
        )

        if self.h_inter_dim > 0:
            # Compute the excitations for interneurons
            exc_cat = [h_pyr, input]
            if self.use_fb:
                exc_cat.append(fb)
            exc_inter = self.pre_inh_activation(
                self.conv_exc_inter(torch.cat(exc_cat, dim=1))
            )
            # Compute the inhibitions
            inh_pyr = self.pre_inh_activation(self.conv_inh(exc_inter))
        else:
            inh_pyr = 0

        # Computer candidate neural memory (cnm) states
        cnm_pyr = self.post_inh_activation(exc_pyr - inh_pyr)

        if self.h_inter_dim > 0:
            cnm_inter = self.post_inh_activation(exc_inter)

        # Euler update for the cell state
        tau_pyr = torch.sigmoid(self.tau_pyr)
        h_next_pyr = (1 - tau_pyr) * h_pyr + tau_pyr * cnm_pyr

        if self.h_inter_dim > 0:
            tau_inter = torch.sigmoid(self.tau_inter)
            h_next_inter = (1 - tau_inter) * h_inter + tau_inter * cnm_inter
        else:
            h_next_inter = None

        # Pool the output
        out = self.out_pool(h_next_pyr)

        return h_next_pyr, h_next_inter, out'''
        
        
class Conv2dEIRNN(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        input_dim: int,
        h_pyr_dim: int | list[int],
        h_inter_dim: int | list[int],
        fb_dim: int | list[int],
        fb_to_inter: bool,
        exc_kernel_size: list[int, int] | list[list[int, int]],
        inh_kernel_size: list[int, int] | list[list[int, int]],
        immediate_inhibition: bool,
        num_layers: int,
        num_steps: int,
        num_classes: Optional[int],
        modulation: bool,
        modulation_type: str,
        modulation_on: str,
        modulation_timestep: str,
        pertubation: bool,
        pertubation_type: str,
        pertubation_on: str,
        pertubation_timestep: int,
        layer_time_delay: bool,
        exc_rectify: Optional[str],
        inh_rectify: Optional[str],
        flush_hidden: bool,
        hidden_init_mode: str,
        fb_init_mode: str,
        out_init_mode: str,
        fb_adjacency: Optional[torch.Tensor],
        pool_kernel_size: list[int, int] | list[list[int, int]],
        pool_stride: list[int, int] | list[list[int, int]],
        bias: bool | list[bool],
        dropout: float,
        pre_inh_activation: Optional[str],
        post_inh_activation: Optional[str],
        fc_dim: int,
        extracted_fb: bool = False,
        extracted_fb_alpha: float = 1.0,
        extracted_fb_apply_to: str = "mixture",
        extracted_fb_timestep: str | int = "all",
        extracted_fb_vector_path: Optional[str] = None,
    ):
        """
        Initialize the Conv2dEIRNN.

        Args:
            input_size (tuple[int, int]): Height and width of input tensor as (height, width).
            input_dim (int): Number of channels of input tensor.
            h_pyr_dim (int | list[int]): Number of channels of the pyramidal neurons or a list of number of channels for each layer.
            h_inter_dims (list[int] | list[list[int]]): Number of channels of the interneurons or a list of number of channels for each layer.
            fb_dim (int | list[int]): Number of channels of the feedback activationsor a list of number of channels for each layer.
            exc_kernel_size (list[int, int] | list[list[int, int]]): Size of the kernel for excitatory convolutions or a list of kernel sizes for each layer.
            inh_kernel_size (list[int, int] | list[list[int, int]]): Size of the kernel for inhibitory convolutions or a list of kernel sizes for each layer.
            num_layers (int): Number of layers in the RNN.
            num_steps (int): Number of iterations to perform in each layer.
            num_classes (int): Number of output classes. If None, the activations of the final layer at the last time step will be output.
            fb_adjacency (Optional[torch.Tensor], optional): Adjacency matrix for feedback connections.
            pool_kernel_size (list[int, int] | list[list[int, int]], optional): Size of the kernel for pooling or a list of kernel sizes for each layer.
            pool_stride (list[int, int] | list[list[int, int]], optional): Stride of the pooling operation or a list of strides for each layer.
            bias (bool | list[bool], optional): Whether or not to add the bias or a list of booleans indicating whether to add bias for each layer.
            activation (str, optional): Activation function to use. Only 'tanh' and 'relu' activations are supported.
            fc_dim (int, optional): Dimension of the fully connected layer.
        """
        super().__init__()
        self.h_pyr_dims = self._extend_for_multilayer(h_pyr_dim, num_layers)
        self.h_inter_dims = self._extend_for_multilayer(
            h_inter_dim, num_layers
        )
        self.fb_dims = self._extend_for_multilayer(fb_dim, num_layers)
        self.exc_kernel_sizes = self._extend_for_multilayer(
            exc_kernel_size, num_layers, depth=1
        )
        self.inh_kernel_sizes = self._extend_for_multilayer(
            inh_kernel_size, num_layers, depth=1
        )
        self.num_steps = num_steps
        self.modulation = modulation
        self.modulation_type = modulation_type
        self.modulation_on = modulation_on
        self.modulation_timestep = modulation_timestep
        self.pertubation = pertubation
        self.pertubation_type = pertubation_type
        self.pertubation_on = pertubation_on
        self.pertubation_timestep = pertubation_timestep
        self.layer_time_delay = layer_time_delay
        self.fb_to_inter = fb_to_inter

        # Fixed, extracted rank-one top-down feedback.
        # The vectors are registered as buffers, so they move with .to(device)
        # but are not trained by the optimizer.
        self.extracted_fb_enabled = bool(extracted_fb)
        self.extracted_fb_alpha = float(extracted_fb_alpha)
        self.extracted_fb_apply_to = extracted_fb_apply_to
        self.extracted_fb_timestep = extracted_fb_timestep
        self._extracted_fb_vector_names: list[str] = []
        if self.extracted_fb_apply_to not in ("mixture", "cue", "all"):
            raise ValueError("extracted_fb_apply_to must be 'mixture', 'cue', or 'all'.")
        if modulation:
            if modulation_type != "lr":
                raise ValueError("modulation_type must be 'lr'")
            if modulation_on not in ("hidden", "layer_output"):
                raise ValueError(
                    "modulation_on must be 'hidden' or 'layer_output'."
                )
            if (
                modulation_timestep != "all"
                and 0 < modulation_timestep < num_steps
            ):
                raise ValueError(
                    "modulation_timestep must be 'all' or an integer between 0 and num_steps."
                )
        if pertubation:
            if pertubation_type != "lr":
                raise ValueError("pertubation_type must be 'lr'.")
            if pertubation_on not in ("hidden", "layer_output"):
                raise ValueError(
                    "pertubation_on must be 'hidden' or 'layer_output'."
                )
            if (
                pertubation_timestep != "all"
                and 0 < pertubation_timestep < num_steps
            ):
                raise ValueError(
                    "modulation_timestep must be 'all' or an integer between 0 and num_steps."
                )
        self.flush_hidden = flush_hidden
        self.hidden_init_mode = hidden_init_mode
        self.fb_init_mode = fb_init_mode
        self.out_init_mode = out_init_mode
        self.pool_kernel_sizes = self._extend_for_multilayer(
            pool_kernel_size, num_layers, depth=1
        )
        self.pool_strides = self._extend_for_multilayer(
            pool_stride, num_layers, depth=1
        )
        self.biases = self._extend_for_multilayer(bias, num_layers)

        self.input_sizes = [input_size]
        for i in range(num_layers):
            self.input_sizes.append(
                (
                    ceil(self.input_sizes[i][0] / self.pool_strides[i][0]),
                    ceil(self.input_sizes[i][1] / self.pool_strides[i][1]),
                )
            )
        self.output_sizes = self.input_sizes[1:]
        self.input_sizes = self.input_sizes[:-1]

        self.use_fb = [False] * num_layers
        self.fb_adjacency = fb_adjacency
        if fb_adjacency is not None:
            try:
                fb_adjacency = torch.load(fb_adjacency)
            except AttributeError:
                fb_adjacency = torch.tensor(fb_adjacency)
            if (
                fb_adjacency.dim() != 2
                or fb_adjacency.shape[0] != num_layers
                or fb_adjacency.shape[1] != num_layers
            ):
                raise ValueError(
                    "The the dimensions of fb_adjacency must match number of layers."
                )
            if fb_adjacency.count_nonzero() == 0:
                raise ValueError(
                    "fb_adjacency must be a non-zero tensor if provided."
                )

            if exc_rectify == "pos":
                Conv2dFb = Conv2dPositive
            elif exc_rectify is None:
                Conv2dFb = nn.Conv2d
            self.fb_adjacency = []
            self.fb_convs = nn.ModuleDict()
            for i, row in enumerate(fb_adjacency):
                row = row.nonzero().squeeze(1).tolist()
                self.fb_adjacency.append(row)
                for j in row:
                    self.use_fb[j] = True
                    upsample = nn.Upsample(
                        size=self.input_sizes[j], mode="bilinear"
                    )
                    conv_exc = Conv2dFb(
                        in_channels=self.h_pyr_dims[i],
                        out_channels=self.fb_dims[j],
                        kernel_size=1,
                        bias=True,
                    )
                    self.fb_convs[f"fb_conv_{i}_{j}"] = nn.Sequential(
                        upsample, conv_exc
                    )

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                Conv2dEIRNNCell(
                    input_size=self.input_sizes[i],
                    input_dim=(
                        input_dim if i == 0 else self.h_pyr_dims[i - 1]
                    ),
                    h_pyr_dim=self.h_pyr_dims[i],
                    h_inter_dim=self.h_inter_dims[i],
                    fb_dim=self.fb_dims[i] if self.use_fb[i] else 0,
                    exc_kernel_size=self.exc_kernel_sizes[i],
                    inh_kernel_size=self.inh_kernel_sizes[i],
                    immediate_inhibition=immediate_inhibition,
                    exc_rectify=exc_rectify,
                    inh_rectify=inh_rectify,
                    pool_kernel_size=self.pool_kernel_sizes[i],
                    pool_stride=self.pool_strides[i],
                    bias=self.biases[i],
                    pre_inh_activation=pre_inh_activation,
                    post_inh_activation=post_inh_activation,
                    fb_to_inter=self.fb_to_inter,
                )
            )

        if pertubation:
            self.pertubations = nn.ModuleList()
            self.pertubations_inter = nn.ModuleList()
            for i in range(num_layers):
                if pertubation_on == "hidden":
                    self.pertubations.append(
                        LowRankPerturbation(
                            self.h_pyr_dims[i],
                            self.input_sizes[i],
                        )
                    )
                    self.pertubations_inter.append(
                        LowRankPerturbation(
                            self.h_inter_dims[i],
                            self.input_sizes[i],
                        )
                    )
                else:
                    self.pertubations.append(
                        LowRankPerturbation(
                            self.h_pyr_dims[i], self.output_sizes[i]
                        )
                    )

            
        if modulation:
            self.modulations = nn.ModuleList()
            self.modulations_inter = nn.ModuleList()
            for i in range(num_layers):
                if modulation_on == "hidden":
                    self.modulations.append(
                        LowRankModulation(
                            self.h_pyr_dims[i],
                            self.input_sizes[i],
                        )
                    )
                    self.modulations_inter.append(
                        LowRankModulation(
                            self.h_inter_dims[i],
                            self.input_sizes[i],
                        )
                    )
                else:
                    self.modulations.append(
                        LowRankModulation(self.h_pyr_dims[i], self.output_sizes[i])
                    )
                    

        self.out_layer = (
            nn.Sequential(
                nn.Flatten(1),
                nn.Linear(
                    self.h_pyr_dims[-1] * prod(self.output_sizes[-1]),
                    fc_dim,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_dim, num_classes),
            )
            if num_classes is not None and num_classes > 0
            else nn.Identity()
        )
        '''self.out_layer = (
        nn.Sequential(
            nn.Conv2d(self.h_pyr_dims[-1], 64, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),   # 8x8 -> 4x4
            nn.Flatten(1),                           # 64*4*4 = 1024
            nn.Linear(64 * 4 * 4, num_classes),
        )
        if num_classes is not None and num_classes > 0
        else nn.Identity()
    )'''


        if extracted_fb_vector_path is not None:
            self.load_extracted_feedback_vectors(
                extracted_fb_vector_path,
                alpha=self.extracted_fb_alpha,
                enabled=self.extracted_fb_enabled,
                map_location="cpu",
            )

    def _init_hidden(self, batch_size, init_mode="zeros", device=None):
        h_pyrs = []
        h_inters = []
        for layer in self.layers:
            h_pyr, h_inter = layer.init_hidden(
                batch_size, init_mode=init_mode, device=device
            )
            h_pyrs.append(h_pyr)
            h_inters.append(h_inter)
        return h_pyrs, h_inters

    def _init_fb(self, batch_size, init_mode="zeros", device=None):
        h_fbs = []
        for layer in self.layers:
            h_fb = layer.init_fb(
                batch_size, init_mode=init_mode, device=device
            )
            h_fbs.append(h_fb)
        return h_fbs

    def _init_out(self, batch_size, init_mode="zeros", device=None):
        outs = []
        for layer in self.layers:
            out = layer.init_out(
                batch_size, init_mode=init_mode, device=device
            )
            outs.append(out)
        return outs

    @staticmethod
    def _extend_for_multilayer(param, num_layers, depth=0):
        inner = param
        for _ in range(depth):
            if not isinstance(inner, Iterable):
                break
            inner = inner[0]

        if not isinstance(inner, Iterable):
            param = [param] * num_layers
        elif len(param) != num_layers:
            raise ValueError(
                "The length of param must match the number of layers if it is a list."
            )
        return param


    def set_extracted_feedback_vectors(
        self,
        vectors: list[torch.Tensor] | tuple[torch.Tensor, ...],
        alpha: Optional[float] = None,
        enabled: bool = True,
    ):
        """Register fixed PCA/SVD feedback vectors for each layer.

        Each vector must have shape [D_l] or [D_l, 1], where
        D_l = h_pyr_dims[l] * input_sizes[l][0] * input_sizes[l][1].
        These tensors are buffers, not Parameters, so they are fixed during training.
        """
        if len(vectors) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} feedback vectors, got {len(vectors)}."
            )

        self._extracted_fb_vector_names = []
        for i, vec in enumerate(vectors):
            vec = torch.as_tensor(vec, dtype=torch.float32).detach().flatten()
            expected_dim = self.h_pyr_dims[i] * prod(self.input_sizes[i])
            if vec.numel() != expected_dim:
                raise ValueError(
                    f"Feedback vector {i} has {vec.numel()} elements, "
                    f"but layer {i} expects {expected_dim}."
                )
            # Unit-normalize to make alpha comparable across layers.
            vec = vec / vec.norm().clamp_min(1e-12)
            name = f"extracted_fb_vector_{i}"
            if hasattr(self, name):
                # Replacing an existing buffer is allowed by assigning the tensor.
                setattr(self, name, vec)
            else:
                self.register_buffer(name, vec, persistent=True)
            self._extracted_fb_vector_names.append(name)

        if alpha is not None:
            self.extracted_fb_alpha = float(alpha)
        self.extracted_fb_enabled = bool(enabled)

    def load_extracted_feedback_vectors(
        self,
        path: str,
        alpha: Optional[float] = None,
        enabled: bool = True,
        map_location: str | torch.device = "cpu",
    ):
        """Load fixed feedback vectors saved by extract_hidden_pca_feedback_vectors."""
        payload = torch.load(path, map_location=map_location)
        vectors = payload["vectors"] if isinstance(payload, dict) and "vectors" in payload else payload
        self.set_extracted_feedback_vectors(vectors, alpha=alpha, enabled=enabled)

    def _has_extracted_feedback_vectors(self) -> bool:
        return len(self._extracted_fb_vector_names) == len(self.layers)

    def _should_apply_extracted_feedback(self, stimulation, cue, mixture, t: int) -> bool:
        if not self.extracted_fb_enabled or not self._has_extracted_feedback_vectors():
            return False
        if self.extracted_fb_apply_to == "mixture" and stimulation is not mixture:
            return False
        if self.extracted_fb_apply_to == "cue" and stimulation is not cue:
            return False
        if self.extracted_fb_timestep != "all" and int(self.extracted_fb_timestep) != t:
            return False
        return True

    def _apply_extracted_feedback_to_low_hidden(
        self,
        low_hidden: torch.Tensor,
        high_hidden: torch.Tensor,
        low_layer_idx: int,
    ) -> torch.Tensor:
        """Apply fixed rank-one feedback from layer low_layer_idx+1 to low_layer_idx.

        If u_high and u_low are the PCA vectors extracted from the trained model,
        then the implicit feedback matrix is R = u_high u_low^T.  We never build R.
        Instead, high @ R = (high @ u_high) u_low^T.
        """
        high_layer_idx = low_layer_idx + 1
        u_low = getattr(self, self._extracted_fb_vector_names[low_layer_idx])
        u_high = getattr(self, self._extracted_fb_vector_names[high_layer_idx])

        low_flat = low_hidden.flatten(1)
        high_flat = high_hidden.flatten(1)

        if low_flat.shape[1] != u_low.numel():
            raise ValueError(
                f"Low layer {low_layer_idx} flattened dim is {low_flat.shape[1]}, "
                f"but feedback vector has {u_low.numel()} elements."
            )
        if high_flat.shape[1] != u_high.numel():
            raise ValueError(
                f"High layer {high_layer_idx} flattened dim is {high_flat.shape[1]}, "
                f"but feedback vector has {u_high.numel()} elements."
            )

        u_low = u_low.to(device=low_hidden.device, dtype=low_flat.dtype).view(1, -1)
        u_high = u_high.to(device=high_hidden.device, dtype=high_flat.dtype).view(-1, 1)

        coef = high_flat @ u_high              # [B, 1]
        fb_low_flat = coef @ u_low             # [B, D_low]
        modulated_low_flat = low_flat * (1.0 + self.extracted_fb_alpha * fb_low_flat)
        return modulated_low_flat.view_as(low_hidden)

    def forward(
        self,
        cue: Optional[torch.Tensor],
        mixture: torch.Tensor,
        all_timesteps: bool = False,
        return_layer_outputs: bool = False,
        return_hidden: bool = False,
        return_phase_hidden: bool = False,
    ):
        """
        Performs forward pass of the Conv2dEIRNN.

        Args:
            cue (torch.Tensor): Input of shape (b, c, h, w) or (b, s, c, h, w), where s is sequence length.
                Used to "prime" the network with a cue stimulus. Optional.
            mixture (torch.Tensor): Input tensor of shape (b, c, h, w) or (b, s, c, h, w), where s is sequence length.
                The primary stimulus to be processed.

        Returns:
            torch.Tensor: Output tensor after pooling of shape (b, n), where n is the number of classes.
        """
        device = mixture.device
        batch_size = mixture.shape[0]

        pertubations_pyr = None
        pertubations_inter = None
        pertubations_out = None
        h_pyrs_cue = None
        h_inters_cue = None
        outs_cue = None
        phase_hidden = {"cue": None, "mixture": None}
        phase_outputs = {"cue": None, "mixture": None}
        for stimulation in (cue, mixture):
            if stimulation is None:
                continue
            if stimulation is cue or cue is None or self.flush_hidden:
                h_pyrs = [
                    [None] * len(self.layers) for _ in range(self.num_steps)
                ]
                h_inters = [
                    [None] * len(self.layers) for _ in range(self.num_steps)
                ]
                h_pyrs[-1], h_inters[-1] = self._init_hidden(
                    batch_size, init_mode=self.hidden_init_mode, device=device
                )
            fbs = [[None] * len(self.layers) for _ in range(self.num_steps)]
            fbs[-1] = self._init_fb(
                batch_size, init_mode=self.fb_init_mode, device=device
            )
            outs = [[None] * len(self.layers) for _ in range(self.num_steps)]
            if self.layer_time_delay:
                outs[-1] = self._init_out(
                    batch_size, init_mode=self.out_init_mode, device=device
                )
            for t in range(self.num_steps):
                if stimulation.dim() == 5:
                    input = stimulation[:, t, ...]
                elif stimulation.dim() == 4:
                    input = stimulation
                else:
                    raise ValueError(
                        "The input must be a 4D tensor or a 5D tensor with sequence length."
                    )
                for i, layer in enumerate(self.layers):
                    # Apply lrp to mixture
                    if (
                        stimulation is mixture
                        and self.pertubation
                        and (
                            self.pertubation_timestep == "all"
                            or self.pertubation_timestep == t
                        )
                    ):
                        if self.pertubation_on == "hidden":
                            h_pyrs[t][i] = h_pyrs[t][i] + pertubations_pyr[i]
                            h_inters[t][i] = (
                                h_inters[t][i] + pertubations_inter[i]
                            )
                        else:
                            outs[t][i] = outs[t][i] + pertubations_out[i]

                    # Compute layer update and output.
                    # Optional extracted feedback is applied BEFORE the cell update
                    # (choice A): high-layer hidden from the previous recurrent
                    # step multiplicatively modulates the low-layer hidden state
                    # that will be used by the current update.
                    h_pyr_for_update = h_pyrs[t - 1][i]
                    if (
                        self._should_apply_extracted_feedback(stimulation, cue, mixture, t)
                        and i < len(self.layers) - 1
                    ):
                        h_pyr_for_update = self._apply_extracted_feedback_to_low_hidden(
                            low_hidden=h_pyr_for_update,
                            high_hidden=h_pyrs[t - 1][i + 1],
                            low_layer_idx=i,
                        )

                    (h_pyrs[t][i], h_inters[t][i], outs[t][i]) = layer(
                        input=(
                            input
                            if i == 0
                            else (
                                outs[t - 1][i - 1]
                                if self.layer_time_delay
                                else outs[t][i - 1]
                            )
                        ),
                        h_pyr=h_pyr_for_update,
                        h_inter=h_inters[t - 1][i],
                        fb=fbs[t - 1][i] if self.use_fb[i] else None,
                    )

                    # Apply modulation to mixture
                    if (
                        stimulation is mixture
                        and self.modulation
                        and (
                            self.modulation_timestep == "all"
                            or self.modulation_timestep == t
                        )
                    ):
                        if self.modulation_on == "hidden":
                            h_pyr_cue = h_pyrs_cue[t][i]
                            h_inter_cue = h_inters_cue[t][i]
                            h_pyrs[t][i] = self.modulations[i](
                                h_pyr_cue, h_pyrs[t][i]
                            )
                            h_inters[t][i] = self.modulations_inter[i](
                                h_inter_cue, h_inters[t][i]
                            )
                        else:
                            out_cue = outs_cue[t][i]
                            outs[t][i] = self.modulations[i](
                                out_cue, outs[t][i]
                            )

                    # Apply feedback
                    if self.fb_adjacency is not None:
                        for j in self.fb_adjacency[i]:
                            if fbs[t][j] is None:
                                fbs[t][j] = self.fb_convs[f"fb_conv_{i}_{j}"](outs[t][i])
                            else:
                                fbs[t][j] += self.fb_convs[f"fb_conv_{i}_{j}"](outs[t][i])

            if self.pertubation and stimulation is cue:
                pertubations_pyr = [0] * len(self.layers)
                pertubations_inter = [0] * len(self.layers)
                pertubations_out = [0] * len(self.layers)
                for i in range(len(self.layers)):
                    if self.pertubation_on == "hidden":
                        pertubations_pyr[i] = self.pertubations[i](h_pyrs[i])
                        pertubations_inter[i] = self.pertubations_inter[i](
                            h_inters[i]
                        )
                    else:
                        pertubations_out[i] = self.pertubations[i](h_pyrs[i])
            if stimulation is cue:
                phase_hidden["cue"] = (h_pyrs, h_inters)
                phase_outputs["cue"] = outs
                outs_cue = outs
                h_pyrs_cue = h_pyrs
                h_inters_cue = h_inters
            elif stimulation is mixture:
                phase_hidden["mixture"] = (h_pyrs, h_inters)
                phase_outputs["mixture"] = outs

        if all_timesteps:
            out = []
            for t in range(self.num_steps):
                out.append(self.out_layer(outs[t][-1]))
        else:
            out = self.out_layer(outs[-1][-1])

        if return_phase_hidden:
            return out, phase_outputs, phase_hidden
        if return_layer_outputs and return_hidden:
            return out, outs, (h_pyrs, h_inters)
        if return_layer_outputs:
            return out, outs
        if return_hidden:
            return out, (h_pyrs, h_inters)
        return out




def extract_hidden_pca_feedback_vectors(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = 1,
    center: bool = True,
    save_path: Optional[str] = None,
    source_phase: str = "cue",
    target_phase: str = "mixture",
):
    """Extract one fixed feedback vector from cross-time covariance per layer.

    This implements the user's intended construction:
        H_l^(t)      = early/source hidden matrix, shape [num_steps * total_B, D_l]
        H_l^(t+T)    = late/target hidden matrix, shape [num_steps * total_B, D_l]
        C_l          = (H_l^(t)).T @ H_l^(t+T), shape [D_l, D_l]

    We then take the rank-1 SVD/PCA direction of C_l. For C_l = U S V^T,
    the saved vector is V[:, 0], i.e. the target-side direction. Because the
    later fixed feedback uses u_{l+1} and u_l to define R_l = u_{l+1} u_l^T,
    this keeps the same downstream interface as before while basing u_l on
    cross-time covariance rather than on a single hidden-state cloud.

    The batch dimension is preserved during flattening:
        h_pyrs[t][l]: [B, C_l, H_l, W_l] -> [B, D_l].
    """
    if source_phase not in ("cue", "mixture"):
        raise ValueError("source_phase must be 'cue' or 'mixture'.")
    if target_phase not in ("cue", "mixture"):
        raise ValueError("target_phase must be 'cue' or 'mixture'.")

    was_training = model.training
    model.eval()
    base_model = getattr(model, "_orig_mod", model)
    num_layers = len(base_model.layers)

    source_rows: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    target_rows: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

    def _unpack(batch):
        if len(batch) == 3:
            cue, mixture, labels = batch
        elif len(batch) == 4:
            cue, mixture, labels, _ = batch
        elif len(batch) == 5:
            cue, mixture, labels, _, _ = batch
        else:
            raise ValueError(f"Unexpected batch structure with {len(batch)} elements")
        return cue, mixture

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            cue, mixture = _unpack(batch)
            cue = cue.to(device, non_blocking=(device.type == "cuda")) if cue is not None else None
            mixture = mixture.to(device, non_blocking=(device.type == "cuda"))

            _, _, phase_hidden = model(
                cue,
                mixture,
                return_phase_hidden=True,
            )
            if phase_hidden[source_phase] is None:
                raise ValueError(f"source_phase={source_phase!r} was not available in this forward pass.")
            if phase_hidden[target_phase] is None:
                raise ValueError(f"target_phase={target_phase!r} was not available in this forward pass.")

            h_pyrs_source, _ = phase_hidden[source_phase]
            h_pyrs_target, _ = phase_hidden[target_phase]

            for layer_idx in range(num_layers):
                src_rows = []
                tgt_rows = []
                for t in range(base_model.num_steps):
                    src = h_pyrs_source[t][layer_idx].detach().float().cpu().flatten(1)
                    tgt = h_pyrs_target[t][layer_idx].detach().float().cpu().flatten(1)
                    src_rows.append(src)
                    tgt_rows.append(tgt)
                source_rows[layer_idx].append(torch.cat(src_rows, dim=0))
                target_rows[layer_idx].append(torch.cat(tgt_rows, dim=0))

    vectors = []
    singular_ratio = []
    for layer_idx in range(num_layers):
        if not source_rows[layer_idx] or not target_rows[layer_idx]:
            raise ValueError("No hidden states were collected. Check num_batches and dataloader.")

        X = torch.cat(source_rows[layer_idx], dim=0)  # [num_steps * total_B, D_l]
        Y = torch.cat(target_rows[layer_idx], dim=0)  # [num_steps * total_B, D_l]
        if X.shape != Y.shape:
            raise ValueError(f"Source/target shape mismatch at layer {layer_idx}: {X.shape} vs {Y.shape}")

        if center:
            X = X - X.mean(dim=0, keepdim=True)
            Y = Y - Y.mean(dim=0, keepdim=True)

        # Cross-time covariance / cross Gram matrix. We normalize by sample count
        # for scale stability, but the singular vectors are unchanged.
        C = (X.T @ Y) / max(1, X.shape[0] - 1)  # [D_l, D_l]

        # SVD of C. Vh[0] is the first target-side right singular vector.
        _, S, Vh = torch.linalg.svd(C, full_matrices=False)
        vec = Vh[0].contiguous()
        vec = vec / vec.norm().clamp_min(1e-12)
        vectors.append(vec.cpu())
        singular_ratio.append(float((S[0] ** 2 / (S ** 2).sum().clamp_min(1e-12)).item()))

    payload = {
        "vectors": vectors,
        "singular_value_ratio_rank1": singular_ratio,
        "h_pyr_dims": list(base_model.h_pyr_dims),
        "input_sizes": list(base_model.input_sizes),
        "num_steps": int(base_model.num_steps),
        "center": bool(center),
        "method": "cross_time_covariance_svd",
        "source_phase": source_phase,
        "target_phase": target_phase,
    }
    if save_path is not None:
        torch.save(payload, save_path)
    if was_training:
        model.train()
    return payload
