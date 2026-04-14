# Run outputs
runs_dir = "runs"

# Dataset
dataset_root = "data/bciciv2a"
data_files = ["**/BCICIV_2a_[1-9].csv"]
dataset_fps = 250.0
train_session_suffixes = ("T",)
train_subject_ids = ("A01", "A02", "A03", "A04", "A05", "A06", "A07")
val_subject_ids = ("A08", "A09")
action_source = "inferred"  # "inferred", "label", or "none"
num_action_classes = 4

# EEG preprocessing
eeg_num_channels = 22
eeg_bandpass_low_hz = 8.0
eeg_bandpass_high_hz = 30.0
eeg_epoch_start_seconds = -0.1
eeg_epoch_end_seconds = 0.7
eeg_patch_size = 25

# Optimization
batch_size = 16
num_workers = 2
persistent_workers = True
prefetch_factor = 4
max_steps = 3000
lr = 3e-4
warmup_steps = 10
weight_decay = 1e-4
sigreg_weight = 0.09
codebook_loss_weight = 1.0
commitment_loss_weight = 1.0

# Logging and checkpointing
checkpoint_every_steps = 1000
metrics_every_steps = 1000

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
