# Check the amount of GPU memory available (we need at least ~16GB)
# CUDA = NVIDIA GPU backend
# MPS = macOS Metal Performance Shaders backend (Apple Silicon)
# Note: Training on macOS/MPS is significantly slower than on CUDA/NVIDIA GPUs.
# MPS is fine for learning, experimentation and inference, but expect longer training times.
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

if DEVICE == "cuda":
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - reserved_memory
    
    print(f"Backend: CUDA")
    print(f"GPU: {gpu_name}")
    print(f"Total Memory:     {total_memory / 1e6:.2f} MB | {total_memory / 1e9:.2f} GB")
    print(f"Allocated Memory: {allocated_memory / 1e6:.2f} MB | {allocated_memory / 1e9:.2f} GB")
    print(f"Reserved Memory:  {reserved_memory / 1e6:.2f} MB | {reserved_memory / 1e9:.2f} GB")
    print(f"Free Memory:      {free_memory / 1e6:.2f} MB | {free_memory / 1e9:.2f} GB")

elif DEVICE == "mps":
    # Note: MPS doesn't expose detailed memory stats like CUDA.
    # Apple Silicon uses unified memory (shared between CPU and GPU).
    # You can check total system memory as a proxy.
    import subprocess
    total_memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    
    print(f"Backend: MPS")
    print(f"Device: Apple Silicon (Metal Performance Shaders)")
    print(f"Total System Memory (unified): {total_memory / 1e6:.2f} MB | {total_memory / 1e9:.2f} GB")
    
    # Verify MPS works with a quick tensor test
    x = torch.tensor([1.0, 2.0]).to("mps")
    print(f"MPS tensor test: {x.device}")

else:
    print("No GPU available (no CUDA or MPS backend found)")