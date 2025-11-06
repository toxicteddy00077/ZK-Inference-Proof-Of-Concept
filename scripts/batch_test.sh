#!/bin/bash

SERVER="http://127.0.0.1:50051"

# Download sample images if they don't exist
download_sample_images() {
    echo "Downloading sample images..."
    
    # Create images directory
    mkdir -p test_images
    
    # Download sample images from the internet
    wget -q -O test_images/cat.jpg "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
    wget -q -O test_images/dog.jpg "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400"
    wget -q -O test_images/bird.jpg "https://images.unsplash.com/photo-1552728089-57bdde30beb3?w=400"
    wget -q -O test_images/car.jpg "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=400"
    wget -q -O test_images/flower.jpg "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=400"
    
    echo "Sample images downloaded to test_images/"
}

# Check if images exist, if not download them
if [ ! -d "test_images" ]; then
    download_sample_images
fi

echo "=== Running Batch Image Classification ==="
echo ""
echo "Test 1: Single image"
cargo run --quiet -- "$SERVER" test_images/cat.jpg

echo ""
echo "=========================================="
echo ""
echo "Test 2: Three images"
cargo run --quiet -- "$SERVER" test_images/cat.jpg test_images/dog.jpg test_images/bird.jpg

echo ""
echo "=========================================="
echo ""
echo "Test 3: Five images (full batch)"
cargo run --quiet -- "$SERVER" test_images/cat.jpg test_images/dog.jpg test_images/bird.jpg test_images/car.jpg test_images/flower.jpg