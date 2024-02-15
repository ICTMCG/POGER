# POGER

Ten Words Only Still Help: Improving Black-Box AI-Generated Text Detection via Proxy-Guided Efficient Re-Sampling

[Preprint](https://arxiv.org/abs/2402.09199)

## Introduction
With the rapidly increasing application of large language models (LLMs), their abuse has caused many undesirable societal problems such as fake news, academic dishonesty, and information pollution. This makes AI-generated text (AIGT) detection of great importance. Among existing methods, white-box methods are generally superior to black-box methods in terms of performance and generalizability, but they require access to LLMs' internal states and are not applicable to black-box settings. In this paper, we propose to estimate word generation probabilities as pseudo white-box features via multiple re-sampling to help improve AIGT detection under the black-box setting. Specifically, we design POGER, a proxy-guided efficient re-sampling method, which selects a small subset of representative words (e.g., 10 words) for performing multiple re-sampling in black-box AIGT detection. Experiments on datasets containing texts from humans and seven LLMs show that POGER outperforms all baselines in macro F1 under black-box, partial white-box, and out-of-distribution settings and maintains lower re-sampling costs than its existing counterparts.

## Code & Data
Stay tuned.

## How to Cite
```
@article{shi2024ten,
  title={{Ten Words Only Still Help: Improving Black-Box AI-Generated Text Detection via Proxy-Guided Efficient Re-Sampling}},
  author={Shi, Yuhui and Sheng, Qiang and Cao, Juan and Mi, Hao and Hu, Beizhe and Wang, Danding},
  journal={arXiv preprint arXiv:2402.09199},
  url={https://arxiv.org/abs/2402.09199},
  year={2024}
}
```
