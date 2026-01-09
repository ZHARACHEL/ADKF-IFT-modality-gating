from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn
import numpy as np

from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
)
from fs_mol.data.dkt import DKTBatch

from fs_mol.utils.gp_utils import ExactGPLayer

import gpytorch
from gpytorch.distributions import MultivariateNormal

#from fs_mol.utils._stateless import functional_call

FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 200


class ModalityGate(nn.Module):
    """
    Modality-level feature selection gate.
    
    Generates scalar weights for each modality (GNN, ECFP, PC-descriptors) based on
    support set statistics. Uses minimal parameters (<100) by computing only L2 norms.
    
    Design:
    - Input: L2 norm statistics (mean, std) per modality → 6 scalars total
    - Output: Sigmoid-activated scalar gate per modality → 3 scalars
    - Parameters: 6×8 + 8 + 8×3 + 3 = 83 parameters
    """
    
    def __init__(self, num_modalities: int = 3, hidden_dim: int = 8):
        """
        Args:
            num_modalities: Number of modalities (default: 3 for GNN+ECFP+PC)
            hidden_dim: Hidden dimension for gate generator (default: 8 for minimal params)
        """
        super().__init__()
        self.num_modalities = num_modalities
        
        # Gate generator: (num_modalities * 2) → hidden → num_modalities
        # Input: [norm_mean_1, norm_std_1, norm_mean_2, norm_std_2, ...]
        # Output: [gate_1, gate_2, gate_3]
        input_dim = num_modalities * 2  # mean + std per modality
        self.gate_generator = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid()  # Gates in (0, 1)
        )
        
    def compute_statistics(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute L2 norm statistics for each modality.
        
        Args:
            features_list: List of [N, D_i] tensors, one per modality
            
        Returns:
            stats: [1, num_modalities * 2] tensor of [mean_norm, std_norm] per modality
        """
        stats = []
        for feat in features_list:
            # Compute L2 norm per sample: [N, D] → [N]
            norms = torch.norm(feat, p=2, dim=1)  # [N]
            
            # Compute statistics: mean and std of norms
            norm_mean = norms.mean().unsqueeze(0)  # [1]
            #norm_std = norms.std().unsqueeze(0)    # [1]
            norm_std = norms.std(unbiased=False).unsqueeze(0) + 1e-6

            stats.extend([norm_mean, norm_std])
        
        # return torch.stack(stats).unsqueeze(0)  # [1, num_modalities * 2]
        stats = torch.cat(stats, dim=0)      # [6]
        return stats.view(1, -1)             # [1, 6]

    
    def forward(self, support_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Generate modality-level gates from support set.
        
        Args:
            support_features: List of support feature tensors, one per modality
            
        Returns:
            gates: [num_modalities] tensor of scalar gates in (0, 1)
        """
        # Compute statistics from support set
        stats = self.compute_statistics(support_features)  # [1, num_modalities * 2]
        
        # Generate gates
        gates = self.gate_generator(stats)  # [1, num_modalities]
        
        return gates.squeeze(0)  # [num_modalities]


@dataclass(frozen=True)
class ADKTModelConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"
    
    # Modality gating configuration:
    use_modality_gating: bool = True
    gating_hidden_dim: int = 8  # Minimal hidden dim for <100 parameters


class ADKTModel(nn.Module):
    def __init__(self, config: ADKTModelConfig):
        super().__init__()
        self.config = config

        # Create GNN if needed:
        if self.config.used_features.startswith("gnn"):
            self.graph_feature_extractor = GraphFeatureExtractor(
                config.graph_feature_extractor_config
            )

        self.use_fc = self.config.used_features.endswith("+fc")

        # Create MLP if needed:
        # if self.use_fc:
        #     self.fc_out_dim = 2048
        #     # Determine dimension:
        #     fc_in_dim = 0
        #     if "gnn" in self.config.used_features:
        #         # fc_in_dim += self.config.graph_feature_extractor_config.readout_config.output_dim
        #         fc_in_dim += self.graph_feature_extractor.output_dim
        #     if "ecfp" in self.config.used_features:
        #         fc_in_dim += FINGERPRINT_DIM
        #     if "pc-descs" in self.config.used_features:
        #         fc_in_dim += PHYS_CHEM_DESCRIPTORS_DIM

        #     self.fc = nn.Sequential(
        #         nn.Linear(fc_in_dim, 2048),
        #         nn.ReLU(),
        #         nn.Linear(2048, self.fc_out_dim),
        #     )

        if self.use_fc:
            self.fc_out_dim = 2048
            # 计算 fc_in_dim: 求和每个启用的模态的维度
            fc_in_dim = 0
            if "gnn" in self.config.used_features:
                fc_in_dim += self.config.graph_feature_extractor_config.readout_config.output_dim
            if "ecfp" in self.config.used_features:
                fc_in_dim += FINGERPRINT_DIM
            if "pc-descs" in self.config.used_features:
                fc_in_dim += PHYS_CHEM_DESCRIPTORS_DIM
            
            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.fc_out_dim),
            )
        
        # Create modality gate if needed:
        if self.config.use_modality_gating:
            num_modalities = sum([
                "gnn" in self.config.used_features,
                "ecfp" in self.config.used_features,
                "pc-descs" in self.config.used_features
            ])
            self.modality_gate = ModalityGate(
                num_modalities=num_modalities,
                hidden_dim=self.config.gating_hidden_dim
            )

        self.__create_tail_GP(kernel_type=self.config.gp_kernel)

        # Always normalize features for numerical stability with GP
        self.normalizing_features = True

    #gp无关参数
    def feature_extractor_params(self):
        fe_params = []
        for name, param in self.named_parameters():
            if not name.startswith("gp_"):
                fe_params.append(param)
        return fe_params

    #gp相关参数
    def gp_params(self):
        gp_params = []
        for name, param in self.named_parameters():
            if name.startswith("gp_"):
                gp_params.append(param)
        return gp_params
        
    # gp参数重置
    def reinit_gp_params(self, gp_input, use_lengthscale_prior=False):

        self.__create_tail_GP(kernel_type=self.config.gp_kernel)

        if self.config.gp_kernel == 'matern' or self.config.gp_kernel == 'rbf' or self.config.gp_kernel == 'RBF':
            median_lengthscale_init = self.compute_median_lengthscale_init(gp_input)
            
            # Safety check: ensure median_lengthscale_init is valid
            if torch.isnan(median_lengthscale_init) or torch.isinf(median_lengthscale_init) or median_lengthscale_init <= 0:
                median_lengthscale_init = torch.tensor(1.0, device=self.device)
            
            if use_lengthscale_prior:
                scale = 0.25
                loc = torch.log(median_lengthscale_init).item() + scale**2 # make sure that mode=median_lengthscale_init
                # Additional safety check for loc
                if not np.isfinite(loc):
                    loc = 0.0  # Default to exp(0) = 1.0 lengthscale
                lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
                self.gp_model.covar_module.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(self.gp_model.covar_module.base_kernel.lengthscale) * median_lengthscale_init

    def __create_tail_GP(self, kernel_type):
        dummy_train_x = torch.ones(64, self.fc_out_dim)
        dummy_train_y = torch.ones(64)

        if self.config.use_ard:
            ard_num_dims = self.fc_out_dim
        else:
            ard_num_dims = None

        if self.config.use_numeric_labels:
            scale = 0.25
            loc = np.log(0.01) + scale**2 # make sure that mode=0.01
            noise_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
        else:
            scale = 0.25
            loc = np.log(0.1) + scale**2 # make sure that mode=0.1
            noise_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
        
        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior).to(self.device)
        self.gp_model = ExactGPLayer(
            train_x=dummy_train_x, train_y=dummy_train_y, likelihood=self.gp_likelihood, 
            kernel=kernel_type, ard_num_dims=ard_num_dims, use_numeric_labels=self.config.use_numeric_labels
        ).to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model).to(self.device)

    def compute_median_lengthscale_init(self, gp_input):
        dist_squared = torch.cdist(gp_input, gp_input) ** 2
        dist_squared = torch.triu(dist_squared, diagonal=1)
        return torch.sqrt(0.5 * torch.median(dist_squared[dist_squared>0.0]))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_batch: DKTBatch, train_loss: bool, predictive_val_loss: bool=False, is_functional_call: bool=False):
        support_features: List[torch.Tensor] = []
        query_features: List[torch.Tensor] = []
        modality_names = []   # 👈 新增：记模态名字

        # Extract features from each modality
        if "gnn" in self.config.used_features:
            support_features.append(self.graph_feature_extractor(input_batch.support_features))
            query_features.append(self.graph_feature_extractor(input_batch.query_features))
            modality_names.append("gnn")   # 👈 新增
        if "ecfp" in self.config.used_features:
            support_features.append(input_batch.support_features.fingerprints.float())
            query_features.append(input_batch.query_features.fingerprints.float())
            modality_names.append("ecfp")  # 👈 新增
        if "pc-descs" in self.config.used_features:
            # 注意：descriptors 可能包含 NaN 和极端值，需要处理
            support_descs = input_batch.support_features.descriptors.float()
            query_descs = input_batch.query_features.descriptors.float()
            
            # Step 1: 使用每列的均值填充 NaN
            col_means = torch.nanmean(support_descs, dim=0)  # [D]
            col_means = torch.nan_to_num(col_means, nan=0.0)
            
            support_nan_mask = torch.isnan(support_descs)
            query_nan_mask = torch.isnan(query_descs)
            
            support_descs = torch.where(support_nan_mask, col_means.unsqueeze(0).expand_as(support_descs), support_descs)
            query_descs = torch.where(query_nan_mask, col_means.unsqueeze(0).expand_as(query_descs), query_descs) #query用support的列平均值填充
            
            # Step 2: 裁剪极端值（防止数值爆炸）
            support_descs = torch.clamp(support_descs, min=-1e6, max=1e6)
            query_descs = torch.clamp(query_descs, min=-1e6, max=1e6)
            
            # Step 3: 标准化（使用 support 集的统计量）
            col_std = support_descs.std(dim=0) + 1e-6  # 防止除零
            support_descs = (support_descs - col_means) / col_std #support标准化，每一列(原值-平均值）/标准差
            query_descs = (query_descs - col_means) / col_std
            
            # Step 4: 再次裁剪标准化后的极端值
            support_descs = torch.clamp(support_descs, min=-10, max=10)
            query_descs = torch.clamp(query_descs, min=-10, max=10)
            
            # Step 5: 处理可能剩余的 NaN/Inf
            support_descs = torch.nan_to_num(support_descs, nan=0.0, posinf=0.0, neginf=0.0)
            query_descs = torch.nan_to_num(query_descs, nan=0.0, posinf=0.0, neginf=0.0)
            
            support_features.append(support_descs)
            query_features.append(query_descs)
            modality_names.append("pc")  # 👈 新增

        # Apply modality-level gating if enabled
        if self.config.use_modality_gating:
            # Compute gates from support set
            gate_weights = self.modality_gate(support_features)  # [num_modalities]
            
            # Apply gates to both support and query features
            support_features_gated = []
            query_features_gated = []

            self.last_gate_weights = {}  # ✅ 初始化

            for i, (support_feat, query_feat) in enumerate(zip(support_features, query_features)):
                gate = gate_weights[i].view(1, 1)  # [1, 1] for broadcasting
                support_features_gated.append(support_feat * gate)
                query_features_gated.append(query_feat * gate)
                # ✅ 修复：在循环内记录每个模态的门控权重
                self.last_gate_weights[modality_names[i]] = gate_weights[i].detach()
            
            support_features = support_features_gated
            query_features = query_features_gated

        # Concatenate features
        support_features_flat = torch.cat(support_features, dim=1)
        query_features_flat = torch.cat(query_features, dim=1)


        # if self.use_fc:
        #     support_features_flat = self.fc(support_features_flat)
        #     query_features_flat = self.fc(query_features_flat)

        if self.use_fc:
            support_features_flat = self.fc(support_features_flat)
            query_features_flat = self.fc(query_features_flat)


        if self.normalizing_features:
            support_features_flat = torch.nn.functional.normalize(support_features_flat, p=2, dim=1)
            query_features_flat = torch.nn.functional.normalize(query_features_flat, p=2, dim=1)

        if self.config.use_numeric_labels:
            support_labels_converted = input_batch.support_numeric_labels.float()
            query_labels_converted = input_batch.query_numeric_labels.float()
        else:
            support_labels_converted = self.__convert_bool_labels(input_batch.support_labels)
            query_labels_converted = self.__convert_bool_labels(input_batch.query_labels)

        # compute train/val loss if the model is in the training mode
        if self.training:
            assert train_loss is not None
            if train_loss: # compute train loss (on the support set)
                if is_functional_call: # return loss directly
                    # Set train data with current support features
                    self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
                    self.gp_model.train()
                    self.gp_likelihood.train()
                    
                    # Debug: check for NaN in features
                    if torch.isnan(support_features_flat).any():
                        print(f"DEBUG: NaN in support_features_flat!")
                    
                    # Directly use mean_module and covar_module to bypass __call__'s input identity check
                    mean_x = self.gp_model.mean_module(support_features_flat)
                    covar_x = self.gp_model.covar_module(support_features_flat)
                    
                    # Debug: check for NaN in GP outputs
                    if torch.isnan(mean_x).any():
                        print(f"DEBUG: NaN in mean_x!")
                    
                    output = MultivariateNormal(mean_x, covar_x)
                    # Compute negative log marginal likelihood
                    logits = -self.mll(output, support_labels_converted)
                    
                    # Debug: check for NaN in logits
                    if torch.isnan(logits).any():
                        print(f"DEBUG: NaN in f_inner logits! logits = {logits}")
                else:
                    self.reinit_gp_params(support_features_flat.detach(), self.config.use_lengthscale_prior)
                    self.gp_model.set_train_data(inputs=support_features_flat.detach(), targets=support_labels_converted.detach(), strict=False)
                    logits = None
            else: # compute val loss (on the query set)
                assert is_functional_call == True
                if predictive_val_loss:
                    # Debug: check for NaN in features
                    if torch.isnan(support_features_flat).any():
                        print(f"DEBUG f_outer: NaN in support_features_flat!")
                    if torch.isnan(query_features_flat).any():
                        print(f"DEBUG f_outer: NaN in query_features_flat!")
                    
                    self.gp_model.eval()
                    self.gp_likelihood.eval()
                    with gpytorch.settings.detach_test_caches(False):
                        self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)
                        # Get predictions
                        pred_dist = self.gp_likelihood(self.gp_model(query_features_flat))
                        # Debug: check for NaN in predictions
                        if torch.isnan(pred_dist.mean).any() or torch.isnan(pred_dist.variance).any():
                            print(f"DEBUG NaN: pred mean has NaN: {torch.isnan(pred_dist.mean).any()}")
                            print(f"DEBUG NaN: pred var has NaN: {torch.isnan(pred_dist.variance).any()}")
                            print(f"DEBUG NaN: pred var min: {pred_dist.variance.min()}, max: {pred_dist.variance.max()}")
                        logits = -pred_dist.log_prob(query_labels_converted)
                        if torch.isnan(logits).any():
                            print(f"DEBUG NaN in logits! logits = {logits}")
                    self.gp_model.train()
                    self.gp_likelihood.train()
                else:
                    self.gp_model.set_train_data(inputs=query_features_flat, targets=query_labels_converted, strict=False)
                    logits = self.gp_model(query_features_flat)
                    logits = -self.mll(logits, self.gp_model.train_targets)

        # do GP posterior inference if the model is in the evaluation mode
        else:
            assert train_loss is None
            self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels_converted, strict=False)

            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(query_features_flat))

        return logits

    def __convert_bool_labels(self, labels):
        # True -> 1.0; False -> -1.0
        return (labels.float() - 0.5) * 2.0
