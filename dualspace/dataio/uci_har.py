"""UCI HAR loader with (B, D, T) tensors and one-hot conditions (B,6).


- Normalize per channel using train stats.
- Optionally downsample/pad to seq_len in config.
"""
# TODO