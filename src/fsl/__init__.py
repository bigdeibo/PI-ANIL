"""Few-shot learning (FSL) module — episode sampling and encoders for quantitative meta-learning.

- episode: N-way K-shot Q-query episode sampler (sample-level / class-level splits)
- models: 1D-CNN / 1D-ResNet encoders + ProtoNet metric head + episode training loop
"""
from .episode import EpisodeSampler, class_counts, filter_rare_classes
from .models import (Conv1Encoder, ResNet1Encoder, ProtoNet, count_params,
                     support_standardize, train_protonet)

__all__ = [
    "EpisodeSampler", "class_counts", "filter_rare_classes",
    "Conv1Encoder", "ResNet1Encoder", "ProtoNet", "count_params",
    "support_standardize", "train_protonet",
]
