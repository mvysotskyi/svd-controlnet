# svd-controlnet

## Inference

**Requirements**  
- NVIDIA GPU with ≥ 16 GB VRAM (tested on RTX 3080 mobile)

**Steps**  
1. Clone the repo:  
   `git clone https://github.com/mvysotskyi/svd-controlnet`
2. Edit `inference.py`:  
   - Set paths to the input image and depth folder  
   - Ensure `controlnet` model folder is present
3. Run the script

---

## Training

**Requirements**  
- NVIDIA GPU with ≥ 24 GB VRAM  
- ≥ 300 GB disk space  
- CUDA 12.4  
- Tested on 1x RTX 3090

> ⚠️ Training with 24 GB VRAM only works with batch size 1.  
> Batch size 2 tested only on RTX A6000.

**Steps**  
1. Clone the repo:  
   `git clone https://github.com/mvysotskyi/svd-controlnet`
2. Download dataset with precomputed depth
3. Edit `configs/train.yaml`  
4. Run `train.py`
