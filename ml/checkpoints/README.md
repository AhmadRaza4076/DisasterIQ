# Place fine-tuned PyTorch checkpoints here (gitignored)
# After AMD training, copy:
#   /results/dmg/checkpoints/best.ckpt -> damage_best.ckpt
#
# Then set in .env:
#   INFERENCE_MODE=pytorch
#   PYTORCH_CHECKPOINT_PATH=D:\AMD\ml\checkpoints\damage_best.ckpt
