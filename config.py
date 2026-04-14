# Run outputs
runs_dir = "runs"

# Dataset
dataset_backend = "cached"  # "cached" or "raw"
dataset_root = "data/maestro-v3.0.0"
dataset_cache_root = "data/maestro_cache"
dataset_train_splits = ("train",)
dataset_val_splits = ("validation",)
action_source = "inferred"  # "inferred", "label", or "none"
num_action_classes = 0

# Audio preprocessing
audio_sample_rate = 48_000
audio_mono = True
audio_num_channels = 1
audio_clip_seconds = 32.0
audio_clip_stride_seconds = 32.0
audio_sequence_length = 4
audio_sequence_stride = 1
audio_patch_samples = 19_200
audio_normalization = "per_clip"

# Optimization
batch_size = 8
grad_accum_steps = 6
num_workers = 2
persistent_workers = True
prefetch_factor = 4
max_steps = 3000
lr = 3e-4
warmup_steps = 10
weight_decay = 1e-4
sigreg_weight = 0.36
codebook_loss_weight = 1.0
commitment_loss_weight = 1.0

# Logging and checkpointing
checkpoint_every_steps = 200
metrics_every_steps = 200

# Runtime
amp = True
compile = True

# Token encoder
latent_dim = 16
num_codes = 8
codebook_beta = 0.25
frame_hidden_dim = 32
frame_depth = 1
frame_heads = 2
frame_mlp_dim = 64
frame_projector_hidden_dim = 32

# Predictor
predictor_hidden_dim = 16
predictor_depth = 1

# Inverse dynamics
id_hidden_dim = 16
id_depth = 1

# Shared temporal transformer settings
heads = 2
dim_head = 8
mlp_dim = 32
dropout = 0.0

# Device
device = "cuda"
