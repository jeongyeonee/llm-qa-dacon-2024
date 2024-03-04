#!/bin/bash

# NUM_GPU="auto"
# DIR_DATA="../data/"
# TRAIN_DATA="train_final_0213.csv"
# MODEL_PATH="OrionStarAI/Orion-14B-Chat-RAG" #OrionStarAI/Orion-14B-Chat / Qwen/Qwen1.5-7B-Chat
# DIR_SAVE="../model/orion_2_rag/"
# TEST_DATA="test_sample.csv" #test_sample.csv
# INSTRUCTION=""
# EPOCH=3
# LEARNING_RATE=2e-5
# BATCH_SIZE=4
# MAX_NEW_TOKENS=400

NUM_GPU="auto"
DIR_DATA="../data/"
TRAIN_DATA="train_final_0213.csv"
MODEL_PATH="Upstage/SOLAR-10.7B-Instruct-v1.0" #OrionStarAI/Orion-14B-Chat / Qwen/Qwen1.5-7B-Chat
DIR_SAVE="../model/solar_rag3/"
TEST_DATA="test_sample.csv" #test_sample.csv
INSTRUCTION="한국어로 3문장 내외로 대답하세요."
EPOCH=2
LEARNING_RATE=2e-5
BATCH_SIZE=4
MAX_NEW_TOKENS=400


CUDA_VISIBLE_DEVICES=0,1 python3 ./run_sft.py --num_gpu "$NUM_GPU" \
                --dir_data "$DIR_DATA" \
                --train_data "$TRAIN_DATA" \
                --model_path "$MODEL_PATH" \
                --dir_save $DIR_SAVE \
                --instruction "$INSTRUCTION"\
                --epoch $EPOCH \
                --learning_rate $LEARNING_RATE\
                --batch_size $BATCH_SIZE\

CUDA_VISIBLE_DEVICES=0,1 python3 ./inference.py --num_gpu "$NUM_GPU" \
                --dir_data "$DIR_DATA" \
                --dir_save $DIR_SAVE \
                --test_data $TEST_DATA \
                --instruction "$INSTRUCTION"\
                --max_new_tokens $MAX_NEW_TOKENS\

# python3 ./evaluate.py --dir_data "$DIR_DATA" \
#                 --train_data "$TRAIN_DATA" \
#                 --model_path "$MODEL_PATH" \
#                 --dir_save $DIR_SAVE \