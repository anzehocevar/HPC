#!/bin/bash

# List of images to benchmark
IMAGES="720x480.png 1024x768.png 1920x1200.png 3840x2160.png 7680x4320.png"
IMGS_DIR="imgs"

# List of block sizes (of single axis)
BLOCK_SIZES="8 16 24 32"

# List of images to benchmark
MODES="basic advanced"

# Number of repetitions per image
repeat=5

# Compile all modes
module load CUDA
for mode in $MODES; do
    make $mode
done

# Output CSV file
OUTPUT_CSV="benchmark_results_block_size.csv"
echo "Hostname: $(hostname)"
echo "Hostname,Image,Mode,BlockSize,TimeMemcpy1,TimeKernel1,TimeKernel2,TimeKernel3,TimeKernel4,TimeMemcpy2,TimeTotal" > "$OUTPUT_CSV"

for mode in $MODES; do
    for block_size in $BLOCK_SIZES; do
        # Loop through each image
        for img in $IMAGES; do
            base_name="${img%.*}"

            total_memcpy1=0
            total_kernel1=0
            total_kernel2=0
            total_kernel3=0
            total_kernel4=0
            total_memcpy2=0
            total_full=0

            echo "Benchmarking $img with mode $mode and block size ${block_size}x${block_size}"

            for run in $(seq 1 $repeat); do
                # Run the program and capture the output
                output=$("./$mode" "${IMGS_DIR}/$img" "out.png" "$block_size" "$block_size")

                # Parse times from output
                time_memcpy1=$(echo "$output" | grep --color=never "Memcpy (RAM -> VRAM)" | grep --color=never -Eo "[0-9]+\.[0-9]+")
                time_kernel1=$(echo "$output" | grep --color=never "Kernel 1" | grep --color=never -Eo "[0-9]+\.[0-9]+")
                time_kernel2=$(echo "$output" | grep --color=never "Kernel 2" | grep --color=never -Eo "[0-9]+\.[0-9]+")
                time_kernel3=$(echo "$output" | grep --color=never "Kernel 3" | grep --color=never -Eo "[0-9]+\.[0-9]+")
                time_kernel4=$(echo "$output" | grep --color=never "Kernel 4" | grep --color=never -Eo "[0-9]+\.[0-9]+")
                time_memcpy2=$(echo "$output" | grep --color=never "Memcpy (VRAM -> RAM)" | grep --color=never -Eo "[0-9]+\.[0-9]+")
                time_full=$(echo "$output" | grep --color=never "Total execution time" | grep --color=never -Eo "[0-9]+\.[0-9]+")

                total_memcpy1=$(echo "$total_memcpy1" + "${time_memcpy1:-0}" | bc -l)
                total_kernel1=$(echo "$total_kernel1" + "${time_kernel1:-0}" | bc -l)
                total_kernel2=$(echo "$total_kernel2" + "${time_kernel2:-0}" | bc -l)
                total_kernel3=$(echo "$total_kernel3" + "${time_kernel3:-0}" | bc -l)
                total_kernel4=$(echo "$total_kernel4" + "${time_kernel4:-0}" | bc -l)
                total_memcpy2=$(echo "$total_memcpy2" + "${time_memcpy2:-0}" | bc -l)
                total_full=$(echo "$total_full + $time_full" | bc -l)
            done

            # Calculate averages
            avg_memcpy1=$(printf "%.3f" $(echo "scale=3; $total_memcpy1 / $repeat" | bc))
            avg_kernel1=$(printf "%.3f" $(echo "scale=3; $total_kernel1 / $repeat" | bc))
            avg_kernel2=$(printf "%.3f" $(echo "scale=3; $total_kernel2 / $repeat" | bc))
            avg_kernel3=$(printf "%.3f" $(echo "scale=3; $total_kernel3 / $repeat" | bc))
            avg_kernel4=$(printf "%.3f" $(echo "scale=3; $total_kernel4 / $repeat" | bc))
            avg_memcpy2=$(printf "%.3f" $(echo "scale=3; $total_memcpy2 / $repeat" | bc))
            avg_full=$(printf "%.3f" $(echo "scale=3; $total_full / $repeat" | bc))

            echo "$img: Mode=$mode, BlockSize=$block_size, Memcpy1=$avg_memcpy1, Kernel1=$avg_kernel1, Kernel2=$avg_kernel2, Kernel3=$avg_kernel3, Kernel4=$avg_kernel4, Memcpy2=$avg_memcpy2, Full=$avg_full"
            echo ""

            # Write to CSV
            echo "$(hostname),$img,$mode,$block_size,$avg_memcpy1,$avg_kernel1,$avg_kernel2,$avg_kernel3,$avg_kernel4,$avg_memcpy2,$avg_full" >> "$OUTPUT_CSV"
        done
    done
done

echo "Done. Results saved to $OUTPUT_CSV."
