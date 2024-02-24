# ReRoPE - Reversed Rotary Positional Embedding

ReRoPE is a simple yet novel augmentation of [Rotary Position Embedding](https://arxiv.org/abs/2104.09864) which is able to reliably and significantly improve the performance of existing context window extension techniques without the need for fine-tuning. The only caveat to ReRoPE is the model needs to be pre-trained with ReRoPE to take adavantage of the enhanced extrapolation behaviour.

## How it works
The idea is simple. Instead of computing the rotary positions for elements of a sequence from left to right, ReRoPE computes the positions from right to left; instead of the first token in your sequence being positon `0` and the last token being position `n` we flip things around so the last token in the sequence is always position `0` and the position index increases the further into the past the token is.

<div style="text-align:center" align=center><br><img src="assets/rope-vs-rerope-head.png" width=50%></div><br>

Why does this work better? For vanillar RoPE when the LM head sits on 'unseen' positions during extrapolation we get a perplexity explosion because the rotations used for the both the queries and keys were never seen during training. But with ReRoPE the these unseen rotations are pushed as far away from the LM head as possible. ReRoPE still suffers from increased perplexity when extrapolating, but the degredation is more graceful than for vanilla RoPE.
