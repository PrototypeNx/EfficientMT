CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/prompts/inference.yaml\
                                           --ckpt_path models/Integrated_Attention/checkpoint.ckpt \
                                           --ref_video_path assets/references/sample_white_tiger.mp4\
                                           --prompt "cat walking on the beach."
