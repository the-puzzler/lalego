data_files = ["factory_001/workers/worker_001/*.tar"]

skip_n = 4
frames_per_window = 32
window_stride = 32
image_size = 224

batch_size = 128
max_steps = 100
lr = 3e-4
warmup_steps = 10
weight_decay = 1e-4
sigreg_weight = 0.09

latent_dim = 256
encoder_hidden_dim = 256
predictor_hidden_dim = 256
encoder_depth = 4
predictor_depth = 4
heads = 8
dim_head = 64
mlp_dim = 512
dropout = 0.0

device = "cuda"
