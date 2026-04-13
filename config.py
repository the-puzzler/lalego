# Run outputs
runs_dir = "runs"

# Dataset
dataset_root = "data/egocentric10k"
data_files = ["factory_032/workers/worker_*/*.tar"]

# Temporal sampling
skip_n = 4
frames_per_window = 32
window_stride = 128 #(frames per window * skip n for non overlapping)

# Image preprocessing
image_size = 224

# Optimization
batch_size = 16
num_workers = 4
persistent_workers = True
prefetch_factor = 4
max_steps = 100
lr = 3e-4
warmup_steps = 10
weight_decay = 1e-4
sigreg_weight = 0.09
codebook_loss_weight = 1.0
commitment_loss_weight = 1.0

# Logging and checkpointing
checkpoint_every_steps = 50
metrics_every_steps = 10

# Runtime
amp = True
compile = True

# Frame encoder
latent_dim = 256
num_codes = 8
codebook_beta = 0.25
frame_patch_size = 14
frame_hidden_dim = 192
frame_depth = 12
frame_heads = 3
frame_mlp_dim = 768

# Predictor
predictor_hidden_dim = 256
predictor_depth = 4

# Inverse dynamics
id_hidden_dim = 256
id_depth = 4

# Shared temporal transformer settings
heads = 8
dim_head = 64
mlp_dim = 512
dropout = 0.0

# Device
device = "cuda"
