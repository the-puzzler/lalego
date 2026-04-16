# Run outputs
runs_dir = "runs"

# Dataset
dataset_backend = "cached"  # "cached" or "raw"
dataset_root = "data/maestro-v3.0.0"
dataset_cache_root = "data/maestro_cache_5s"
dataset_train_splits = ("train",)
dataset_val_splits = ("validation",)
action_source = "inferred"  # "inferred", "label", or "none"
num_action_classes = 0

# Audio preprocessing
audio_sample_rate = 16_000
audio_mono = True
audio_num_channels = 1
audio_clip_seconds = 5.0
audio_clip_stride_seconds = 5.0
audio_sequence_length = 4
audio_sequence_stride = 1
audio_patch_samples = 6_000
audio_normalization = "none"

# Optimization
batch_size = 12
grad_accum_steps = 6
num_workers = 16
persistent_workers = True
prefetch_factor = 1
dataset_max_cached_payloads = 1
max_steps = 10_000
lr = 3e-4
warmup_steps = 10
weight_decay = 1e-4
sigreg_weight = 0.18
codebook_loss_weight = 0.1#1.0
commitment_loss_weight = 0.1#1.0

# Logging and checkpointing
checkpoint_every_steps = 500
metrics_every_steps = 500

# Runtime
amp = True
compile = True

# Token encoder
frame_encoder_type = "encodec"  # "waveform" or "encodec"
encodec_model_name = "facebook/encodec_24khz"
freeze_encodec = True
latent_dim = 128
num_codes = 8
codebook_beta = 0.25
frame_hidden_dim = 256
frame_depth = 6
frame_heads = 8
frame_mlp_dim = 1_024
frame_projector_hidden_dim = 512

# Predictor
predictor_hidden_dim = 256
predictor_depth = 6

# Inverse dynamics
id_hidden_dim = 256
id_depth = 6

# Shared temporal transformer settings
heads = 8
dim_head = 32
mlp_dim = 1_024
dropout = 0.0

# Device
device = "cuda"
