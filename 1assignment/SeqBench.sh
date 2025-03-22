#!/bin/bash

# List of images to benchmark
images=("720x480.png" "1024x768.png" "1920x1200.png" "3840x2160.png" "7680x4320.png")

# Number of repetitions per image
repeat=5

# Output CSV file
output_csv="benchmark_sequential_results.csv"
echo "Image,AvgEnergyTime,AvgSeamTime" > "$output_csv"

# Loop through each image
for img in "${images[@]}"; do
    base_name="${img%.*}"

    total_energy=0
    total_seam=0

    echo "Benchmarking $img..."

    for run in $(seq 1 $repeat); do
        output_img="out_${base_name}_run${run}.png"

        # Run the program and capture the output
        output=$(./SequentialSeam "$img" "$output_img")

        # Parse times from output
        energy_time=$(echo "$output" | grep "Energy calculation took" | awk '{print $4}')
        seam_time=$(echo "$output" | grep "Seam carving took" | awk '{print $4}')

        total_energy=$(echo "$total_energy + $energy_time" | bc -l)
        total_seam=$(echo "$total_seam + $seam_time" | bc -l)
    done

    # Calculate averages
    avg_energy=$(echo "scale=6; $total_energy / $repeat" | bc)
    avg_seam=$(echo "scale=6; $total_seam / $repeat" | bc)

    echo "$img: Energy=$avg_energy s, Seam=$avg_seam s"

    # Write to CSV
    echo "$img,$avg_energy,$avg_seam" >> "$output_csv"
done

echo "Done. Results saved to $output_csv."
