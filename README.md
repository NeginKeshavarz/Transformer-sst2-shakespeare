# Transformer Pretraining & Sentiment Fine-Tuning

This project pretrains a small Transformer (character-level) on Shakespeare, then fine-tunes on SST2 sentiment analysis using the MMLO loss.

- **Initialization**: Glorot/Xavier
- **Pretraining**: Shakespeare
- **Fine-tuning**: SST2, MMLO loss
- **Model**: Character-level Transformer

## How to run
1. Put your data in `data/input.txt` and `data/train.txt`.
2. Run the main script:
    ```
    python Nanogpt-Task2.py
    ```

## Results
- Example generated text after pretraining:
    ```
    <paste a sample here!>
    ```
- Fine-tuning accuracy: ~1.0 (train set)
