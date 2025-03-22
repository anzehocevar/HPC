#!/bin/bash

# List of images to benchmark
images="720x480.png 1024x768.png 1920x1200.png 3840x2160.png 7680x4320.png"

# Number of repetitions per image
repeat=5

# Output CSV file
mkdir -p out
output_csv="benchmark_sequential_results.csv"
echo "Image,AvgEnergyTime,AvgIdentificationTime,AvgRemovalTime,AvgCopyTime" > "$output_csv"

# Loop through each image
for img in $images; do
    base_name="${img%.*}"

    total_energy=0
    total_identification=0
    total_removal=0
    total_copy=0

    echo "Benchmarking $img..."

    for run in $(seq 1 $repeat); do
        output_img="out/out_${base_name}_run${run}.png"

        # Run the program and capture the output
        output=$(./SequentialSeam "$img" "$output_img")

        # Parse times from output
        energy_time=$(echo "$output" | grep "Energy calculation took" | grep --color=never -E -o "[0-9]+\.[0-9]+")
        identification_time=$(echo "$output" | grep "Vertical seam identification took" | grep --color=never -E -o "[0-9]+\.[0-9]+")
        removal_time=$(echo "$output" | grep "Seam removal took" | grep --color=never -E -o "[0-9]+\.[0-9]+")
        copy_time=$(echo "$output" | grep "Copying took" | grep --color=never -E -o "[0-9]+\.[0-9]+")

        total_energy=$(echo "$total_energy + $energy_time" | bc -l)
        total_identification=$(echo "$total_identification + $identification_time" | bc -l)
        total_removal=$(echo "$total_removal + $removal_time" | bc -l)
        total_copy=$(echo "$total_copy + $copy_time" | bc -l)
    done

    # Calculate averages
    avg_energy=$(printf "%.3f" $(echo "scale=3; $total_energy / $repeat" | bc))
    avg_identification=$(printf "%.3f" $(echo "scale=3; $total_identification / $repeat" | bc))
    avg_removal=$(printf "%.3f" $(echo "scale=3; $total_removal / $repeat" | bc))
    avg_copy=$(printf "%.3f" $(echo "scale=3; $total_copy / $repeat" | bc))

    echo "$img: Energy=$avg_energy s, Identification=$avg_identification s, Removal=$avg_removal s, Copy=$avg_copy s"

    # Write to CSV
    echo "$img,$avg_energy,$avg_identification,$avg_removal,$avg_copy" >> "$output_csv"
done

echo "Done. Results saved to $output_csv."
