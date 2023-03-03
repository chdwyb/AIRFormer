## Frequency-Oriented Efficient Transformer for All-in-One Adverse-Weather Image Restoration
**ABSTRACT**: Adverse weather conditions, such as rain, raindrop, snow and haze, always degrade image in an unpredictable manner, thus most existing task-specific and task-aligned methods have shown incompetence for this challenging problem. To this end, we first explore the general principle of Transformer from the perspective of frequency changing, and propose an efficient frequency-oriented AIRFormer for all-in-one adverse weather removal. Specifically, we discover that the early self-attention mechanism exhibits distinct low-pass-like properties, and accordingly establish a frequency-guided Transformer en- coder utilizing wavelet supplementary prior to guide the feature extraction. Based on the unspecific frequency characteristics of self attention in the late stage, we further develop a frequency-refined Transformer decoder embedded the learnable task-biased queries in the spatial dimension, channel dimension and wavelet domain. In addition, we compose an all-in-one dataset named AIR40K to train the proposed method for adverse-weather image restoration as benchmark. Extensive experiments show that the proposed AIRFormer achieves significant improvements over both task-aligned and all-in-one methods on 15 public datasets. In particular, AIRFormer outperforms AirNet and TransWeather while takes only 9.52% and 34.0% of their inference time respectively. Source code and pre-trained models are available at https://github.com/chdwyb/AIRFormer.



Currently this repository is ***still*** being prepared, further details to be included soon.



