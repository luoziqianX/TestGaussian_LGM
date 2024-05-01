# Control Net for LGM model

Add a control net for **LGM** UNet  
Coming soon

## Install

```bash
conda create -n controled_lgm python=3.10
conda activate controled_lgm
pip install -r require1.txt
pip install -r require2.txt
pip install ./diff-gaussian-rasterization_LGM
pip install git+https://github.com/NVlabs/nvdiffrast
pip install transformers==4.40.1
```

## Pretrained Weights

### For LGM model

You can download the pretrained weights for LGM model from [huggingface](https://huggingface.co/ashawkey/LGM)
For example,to download the fp16 LGM model for inference:

```bash
mkdir pretrained && cd pretrained
wget https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors
cd ..
```

For [MVDream](https://github.com/bytedance/MVDream)
and [ImageDream](https://github.com/bytedance/ImageDream),
we use a [diffuser implementation](https://github.com/ashawkey/mvdream_diffusers).
The weight will be downloaded automatically.

### For Control LGM model

You can make Control LGM model weight from LGM 's pretrained model
by running the following command:

```bash
python make_control_lgm.py --lgm_path LGM_MODEL_PATH --control_lgm_path CONTROL_LGM_MODEL_PATH
```

## Inference

You can run the following command to test the LGM model:

```bash
python baseline_test.py --input_dir INPUT_DIR --device \
 RUNNING_DEVICE --checkpoint_path THE_PATH_OF_LGM_MODEL \
 --output_dir OUTPUT_DIR
```

You can also run the following command to test the Control LGM model:

```bash
python test_control.py --input_dir INPUT_DIR --device \
 RUNNING_DEVICE --checkpoint_path THE_PATH_OF_CONTROL_LGM_MODEL \
 --output_dir OUTPUT_DIR --num_frames NUM_FRAMES_FOR_REBUILD
```

## Dataset
We make the dataset from [Thuman2.1](https://github.com/ytrock/THuman2.0-Dataset).
The making method will **coming soon**.

## Training
**Coming soon!!!**

You can training by running the main.py.
