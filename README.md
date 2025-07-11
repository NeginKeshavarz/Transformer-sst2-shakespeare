# Transformer Pretraining & Sentiment Fine-Tuning

This project pretrains a small Transformer (character-level) on Shakespeare, then fine-tunes on SST2 sentiment analysis then we apply RLHF and PPO training.

- **Initialization**: Glorot/Xavier
- **Pretraining**: Shakespeare
- **Fine-tuning**: SST2
- **Model**: Character-level Transformer

## How to run
1. Put your data in `data/input.txt` and `data/train.txt`.
2. Run the main script:
    ```
    python Nanogpt-Task2.py
    ```

- Fine-tuning accuracy: 0.8550
