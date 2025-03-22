#!/bin/bash

# List of images to benchmark
images="720x480.png 1024x768.png 1920x1200.png 3840x2160.png 7680x4320.png"

num_threads_options="1 2 4 8 16"
num_seams_per_run=128

# Number of repetitions per image
repeat=5

# Output CSV file
mkdir -p out
output_csv="benchmark_results.csv"
echo "Hostname: $(hostname)"
echo "Hostname,Image,NumThreads,AvgEnergyTime,AvgIdentificationTime,AvgRemovalTime,AvgCopyTime,AvgFullTime" > "$output_csv"

# Do everything for all num_threads_options
for num_threads in $num_threads_options; do
    # Loop through each image
    for img in $images; do
        base_name="${img%.*}"

        total_energy=0
        total_identification=0
        total_removal=0
        total_copy=0
        total_full=0

        echo "Benchmarking $img with $num_threads thread(s)..."

        for run in $(seq 1 $repeat); do
            output_img="out/out_${base_name}_run${run}.png"

            # Run the program and capture the output
            output=$(./SeamCarving "$img" "$output_img" "$num_seams_per_run" "$num_threads")

            # Parse times from output
            energy_time=$(echo "$output" | grep "Energy calculation took" | grep --color=never -E -o "[0-9]+\.[0-9]+")
            identification_time=$(echo "$output" | grep "Vertical seam identification took" | grep --color=never -E -o "[0-9]+\.[0-9]+")
            removal_time=$(echo "$output" | grep "Seam removal took" | grep --color=never -E -o "[0-9]+\.[0-9]+")
            copy_time=$(echo "$output" | grep "Copying took" | grep --color=never -E -o "[0-9]+\.[0-9]+")
            full_time=$(echo "$output" | grep "Total time" | grep --color=never -E -o "[0-9]+\.[0-9]+")

            total_energy=$(echo "$total_energy + $energy_time" | bc -l)
            total_identification=$(echo "$total_identification + $identification_time" | bc -l)
            total_removal=$(echo "$total_removal + $removal_time" | bc -l)
            total_copy=$(echo "$total_copy + $copy_time" | bc -l)
            total_full=$(echo "$total_full + $full_time" | bc -l)
        done

        # Calculate averages
        avg_energy=$(printf "%.3f" $(echo "scale=3; $total_energy / $repeat" | bc))
        avg_identification=$(printf "%.3f" $(echo "scale=3; $total_identification / $repeat" | bc))
        avg_removal=$(printf "%.3f" $(echo "scale=3; $total_removal / $repeat" | bc))
        avg_copy=$(printf "%.3f" $(echo "scale=3; $total_copy / $repeat" | bc))
        avg_full=$(printf "%.3f" $(echo "scale=3; $total_full / $repeat" | bc))

        echo "$img: NumThreads=$num_threads, Energy=$avg_energy s, Identification=$avg_identification s, Removal=$avg_removal s, Copy=$avg_copy s, Full=$avg_full s"

        # Write to CSV
        echo "$(hostname),$img,$num_threads,$avg_energy,$avg_identification,$avg_removal,$avg_copy,$avg_full" >> "$output_csv"
    done
done

echo "Done. Results saved to $output_csv."
