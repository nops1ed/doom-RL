#!/bin/bash

if ! pip list | grep -q torch; then
    echo "pip install -r requirements.txt"
    pip install -r requirements.txt
fi

mkdir -p checkpoints logs

echo ""
echo "1. Train (100 episodes)"
echo "2. Train (1000 episodes)"
echo "3. Test (5 episodes)"
echo "4. TensorBoard"
echo "5. Exit"
echo ""
read -p "Select (1-5): " choice
case $choice in
    1)
        echo "Training (100 episodes)..."
        python train.py --episodes 100 --save-dir checkpoints --log-dir logs
        ;;
    2)
        echo "Training (1000 episodes)..."
        python train.py --episodes 1000 --save-dir checkpoints --log-dir logs --render
        ;;
    3)
        if [ -f "checkpoints/best_model.pth" ]; then
            echo "Testing model..."
            python test.py --model checkpoints/best_model.pth --episodes 5 --render
        else
            echo "No model found. Please train first."
        fi
        ;;
    4)
        echo "Starting TensorBoard..."
        if command -v tensorboard &> /dev/null; then
            tensorboard --logdir logs --port 6006
        else
            echo "plz run it first: pip install tensorboard"
        fi
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please enter a number between 1 and 5."
        exit 1
        ;;
esac