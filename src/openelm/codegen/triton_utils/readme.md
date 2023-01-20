## Triton and Faster transformer
[Triton Inference Server](https://github.com/triton-inference-server/server) is an open source inference serving software that streamlines AI inferencing. Triton enables teams to deploy any AI model from multiple deep learning and machine learning frameworks .

[FasterTransformer](https://github.com/NVIDIA/FasterTransformer) is built on top of CUDA, cuBLAS, cuBLASLt and C++, providing highly optimized transformer layers for both the encoder and decoder for inference.
## How to use FT_converter

FT converter converts a Codegen based model to a gptj architecture and further converts it to FT transformer weights


```python convert_ft.py --model_name='Salesforce/codegen-2B-mono --n_gpus=8 --output_dir=" " ```

In AWS cluster please use anything else than /fsx for output_dir=''. Likely ```/admin/home-your-username-here/path-to-dir```

## How to start up the triton server

``` sh triton_start.sh  /admin/home-your-username-here/path-to-dir/model_name```

## Example: Conversion and triton start up

Model conversion

```python convert_ft.py --model_name='Salesforce/codegen-2B-mono --n_gpus=8 --output_dir=/adminn/home-harrysaini/elm_ft/ ```

Triton Startup:

``` sh triton_start.sh /admin/home-harrysaini/elm_ft/codegen-2B-mono-8gpu ```

You should see the triton server starting up; It'll open three ports 8000,8001,8002

8001 is for grpc on which our client is based.

## Example 2: Conversion using one gpu

```python convert_ft.py --model_name='Salesforce/codegen-2B-mono --n_gpus=1 --output_dir=/adminn/home-harrysaini/elm_ft/ ```

Triton server:
Please add SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 in front of command in singularity; such like content of triton_start.sh will be:

````
MODEL_PATH=${1}
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity run --nv /fsx/elm_ft/triton_launch.sif/ mpirun -n 1 tritonserver --model-repository=${MODEL_PATH}
````

Then execute something like this:


```sh triton_start.sh /admin/home-harrysaini/elm_ft/codegen-2B-mono-1gpu```

Though this can give error with singularity.

## Notes:
We've a converted set of models in /fsx/elm_ft/ , but if you'll copy them make sure you change the /fsx/elm_ft/model_name/fastertransformer/config.pbtxt. You'll need to change the checkpoint path to the new path where you intend to copy them.

Or you can just convert them locally.
