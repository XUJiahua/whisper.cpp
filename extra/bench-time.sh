#!/bin/bash

# Helper script to run the bench tool on all models and print the results in share-able format

printf "Usage: ./bench-time.sh wav_filepath [greedy]\n"

# use greedy decoder to speed up
# ./extra/bench-time.sh samples/gb0-00.00.00.000-00.00.30.052.wav greedy
# use beam search decoder, slower
# ./extra/bench-time.sh samples/gb0-00.00.00.000-00.00.30.052.wav

wav_filepath=$1

# by default use greedy decoder
# use greedy decoder to speed up decode time: https://github.com/ggerganov/whisper.cpp/issues/2241
greedy="-bs 1"
if [ -z "$2" ]; then
    greedy=""
fi

models=(                                                                                                    \
      "tiny"     "tiny-q4_0"     "tiny-q4_1"     "tiny-q5_0"     "tiny-q5_1"     "tiny-q8_0"                \
      "base"     "base-q4_0"     "base-q4_1"     "base-q5_0"     "base-q5_1"     "base-q8_0"                \
     "small"    "small-q4_0"    "small-q4_1"    "small-q5_0"    "small-q5_1"    "small-q8_0"                \
    "medium"   "medium-q4_0"   "medium-q4_1"   "medium-q5_0"   "medium-q5_1"   "medium-q8_0"   "medium-dis" \
  "large-v2" "large-v2-q4_0" "large-v2-q4_1" "large-v2-q5_0" "large-v2-q5_1" "large-v2-q8_0" "large-v2-dis" \
)

printf "| %6s | %6s | %16s | %13s | %3s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s |\n" "CPU" "OS" "Config" "Model" "Th" "Load" "Enc." "Dec." "Dec.Run" "Bch5" "PP" "Total" "Mem." "Commit"
printf "| %6s | %6s | %16s | %13s | %3s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s |\n" "---" "---" "---" "---" "---" "---" "---" "---" "---" "---" "---" "---" "---" "---"

for model in "${models[@]}"; do
    # actual run
    # store stderr output in a variable in order to parse it later
    output=$(./main -m ./models/ggml-$model.bin -f $wav_filepath $greedy 2>&1)
    # output=$(./main -m ./models/ggml-$model.bin -f samples/gb0-00.00.00.000-00.00.30.052.wav -bs 1 2>&1)
    ret=$?

    # parse the output:
    load_time=$(echo "$output" | grep "load time" | awk '{print $5}')
    encode_time=$(echo "$output" | grep "encode time" | awk '{print $5}')
    decode_time=$(echo "$output" | grep "decode time" | awk '{print $5}')
    decode_times=$(echo "$output" | grep "decode time" | awk '{print $8}')
    batchd_time=$(echo "$output" | grep "batchd time" | awk '{print $5}')
    prompt_time=$(echo "$output" | grep "prompt time" | awk '{print $5}')
    total_time=$(echo "$output" | grep "total time" | awk '{print $5}')
    system_info=$(echo "$output" | grep "system_info")
    n_threads=$(echo "$output" | grep "system_info" | awk '{print $4}')
    mem_usage=$(echo "$output" | grep "ggml_backend_metal_buffer_type_alloc_buffer" | tail -n 1 | awk '{print $9}')
    # echo $mem_usage

    # floor to milliseconds
    #encode_time=${encode_time%.*}
    #decode_time=${decode_time%.*}
    #prompt_time=${prompt_time%.*}

    config=""

    if [[ $system_info == *"AVX2 = 1"* ]]; then
        config="$config AVX2"
    fi

    if [[ $system_info == *"NEON = 1"* ]]; then
        config="$config NEON"
    fi

    if [[ $system_info == *"BLAS = 1"* ]]; then
        config="$config BLAS"
    fi

    if [[ $system_info == *"COREML = 1"* ]]; then
        config="$config COREML"
    fi

    if [[ $system_info == *"CUDA = 1"* ]]; then
        config="$config CUDA"
    fi

    if [[ $system_info == *"METAL = 1"* ]]; then
        config="$config METAL"
    fi

    commit=$(git rev-parse --short HEAD)

    if [ $ret -eq 0 ]; then
        printf "| <todo> | <todo> | %16s | %13s | %3s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %7s |\n" "$config" "$model" "$n_threads" "$load_time" "$encode_time" "$decode_time" "$decode_times" "$batchd_time" "$prompt_time" "$total_time" "$mem_usage" "$commit"
    fi
done
