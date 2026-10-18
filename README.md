# GE-PEFT GATED EXPANDABLE PARAMETER-EFFICIENT FINE-TUNING FOR CONTINUAL LEARNING

This is the implementation code for the paper `GE-PEFT: Gated Expandable Parameter-Efficient Fine-Tuning for Continual Learning`.

If you use this code in your experiment, consider citing the following paper:
```
Omeliyanenko, J., Hotho, A., & Schlör, D. (2026). GE-PEFT: Gated Expandable Parameter-Efficient Fine-Tuning for Continual Learning. Machine Learning, 115(7), 171.
```

Direct paper link: <https://link.springer.com/article/10.1007/s10994-026-07104-z>

## Contents

The datasets and main methods for training specific models can be found in the `src` folder.

## Running

Requirements are und `requirements.txt`.

Scripts to run experiments with various models are available in the `scripts` folder.

For example, to start GE-PEFT experiments with the Llama 2 language model, run the `GEPEFT_mask_exp_peft_orders_llama.sh` script.
