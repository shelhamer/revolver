# Few-shot Segmentation Propagation with Guided Networks

on arxiv: https://arxiv.org/abs/1806.07373

by Kate Rakelly\*, Evan Shelhamer\*, Trevor Darrell, Alexei A. Efros, and Sergey Levine
UC Berkeley

> Learning-based methods for visual segmentation have made progress on particular
types of segmentation tasks, but are limited by the necessary supervision, the
narrow definitions of fixed tasks, and the lack of control during inference for
correcting errors. To remedy the rigidity and annotation burden of standard
approaches, we address the problem of few-shot segmentation: given few image
and few pixel supervision, segment any images accordingly. We propose guided
networks, which extract a latent task representation from any amount of
supervision, and optimize our architecture end-to-end for fast, accurate
few-shot segmentation. Our method can switch tasks without further optimization
and quickly update when given more guidance. We report the first results for
segmentation from one pixel per concept and show real-time interactive video
segmentation. Our unified approach propagates pixel annotations across space
for interactive segmentation, across time for video segmentation, and across
scenes for semantic segmentation. Our guided segmentor is state-of-the-art in
accuracy for the amount of annotation and time.

This is a **work-in-progress**, not yet a reference implementation of the paper, and could change at any time.

- for few-shot interactive image segmentation and few-shot semantic segmentation, see this branch.
- for few-shot video object segmentation, see branch `video-seg` (note: this is older code, for pytorch 0.3.1)

TODO

- [x] port to pytorch 1.0
- [ ] push branch for interactive video segmentation
- [ ] reconcile branches into unified implementation of few-shot segmentation

Please check back soon for improvements, pre-trained models, and usage notebooks!
