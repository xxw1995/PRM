运行训练(Training)  
⚠️ 运行训练之前，请修改 train/mat/scripts/train_llm.sh 文件中的 $dataset_path, $model_name_or_path 和 $prm_name_or_path 项。  
cd train/mat/scripts  
bash train_llm.sh  


运行PRM学习
cd prm/code  
python finetune_qwen_single_gpu.py --model_path $YOUR_MODEL_PATH \  
                                   --train_data_path $TRAIN_DATA_PATH \  
                                   --test_data_path $TEST_DATA_PATH  

