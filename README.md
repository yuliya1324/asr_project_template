# ASR project

## Installation guide

```shell
pip install -r ./requirements.txt
mkdir default_test_model
cd default_test_model
gdown https://drive.google.com/uc?id=1XJvUUqJ7m1D594S_pa2zw_5YUwYNAzFC
gdown https://drive.google.com/uc?id=1jrZDzYcw32DF9jVTG82hQyR2Yl-1sUp3
cd ..
mkdir lm
wget http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz
wget http://www.openslr.org/resources/11/librispeech-vocab.txt
python prepare_lm.py
```

If command `gdown https://drive.google.com/uc?id=1XJvUUqJ7m1D594S_pa2zw_5YUwYNAzFC` doesn't work (as the file is too big), you can download checkpoints manualy through the following [link](https://drive.google.com/file/d/1XJvUUqJ7m1D594S_pa2zw_5YUwYNAzFC/view?usp=sharing)


## Run test.py

   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json \
      -b 5
   ```

Here is my [report](https://wandb.ai/julia_kor/asr_project/reports/ASR-project--VmlldzoyODAwMjQ2?accessToken=3ivolool42dfd2tjqavr3am2hl72fegyx45f89p0i8aotx30gcnmdc0zbsyraihp)