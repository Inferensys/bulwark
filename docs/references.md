# References and acknowledgements

Bulwark builds on other people's models, code and datasets. If you use Bulwark in
research, please also cite the work below that your use depends on. Each entry is copied
from the citation its authors publish (project README, model card or arXiv).

Dataset citations, licences and access terms are listed per dataset in
[datasets.md](datasets.md). Cite every dataset that a model you use was trained on; the
model card of each weights release lists them.

To cite Bulwark itself, use [CITATION.cff](../CITATION.cff) (GitHub shows it as "Cite this
repository").

## Encoder: V-JEPA 2 and V-JEPA 2.1 (Meta)

Bulwark-B1 uses the V-JEPA 2.1 ViT-B/16 encoder and Meta's model code
(https://github.com/facebookresearch/vjepa2). Citations from the V-JEPA 2 README:

```bibtex
@article{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and
Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and
Arnaud, Sergio and Gejji, Abha and Martin, Ada and Robert Hogan, Francois and Dugas, Daniel and
Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and
Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and
Rabbat, Michael and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```

```bibtex
@article{murlabadia2026vjepa2_1,
  title={V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning},
  author={Mur-Labadia, Lorenzo and Muckley, Matthew and Bar, Amir and Assran, Mahmoud and
Sinha, Koustuv and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas and Bardes, Adrien},
  journal={arXiv preprint arXiv:2603.14482},
  year={2026}
}
```

## Edge backbone: EfficientViT (MIT Han Lab) and timm

Bulwark-E1 uses EfficientViT-B1 (https://github.com/mit-han-lab/efficientvit) through
timm's `efficientvit_b1.r224_in1k` weights. Citations from the EfficientViT README and the
timm README:

```bibtex
@inproceedings{cai2023efficientvit,
  title={Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction},
  author={Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17302--17313},
  year={2023}
}
```

```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

## Teacher models

Teacher models label training clips; they are not part of B1 or E1.

MediaPipe Face Landmarker (Google, https://ai.google.dev/edge/mediapipe), cited through
the MediaPipe framework paper (BibTeX from arXiv):

```bibtex
@misc{lugaresi2019mediapipeframeworkbuildingperception,
      title={MediaPipe: A Framework for Building Perception Pipelines},
      author={Camillo Lugaresi and Jiuqiang Tang and Hadon Nash and Chris McClanahan and Esha Uboweja and Michael Hays and Fan Zhang and Chuo-Ling Chang and Ming Guang Yong and Juhyun Lee and Wan-Teh Chang and Wei Hua and Manfred Georg and Matthias Grundmann},
      year={2019},
      eprint={1906.08172},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/1906.08172},
}
```

RF-DETR (Roboflow, https://github.com/roboflow/rf-detr), citation from its README:

```bibtex
@inproceedings{robinson2026rfdetr,
  title     = {RF-DETR: Real-Time Detection Transformer},
  author    = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei and Ramanan, Deva and Peri, Neehar},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2511.09554}
}
```

Qwen3-VL (Qwen team, https://github.com/QwenLM/Qwen3-VL), citation from its README:

```bibtex
@article{Qwen3-VL,
      title={Qwen3-VL Technical Report},
      author={Shuai Bai and Yuxuan Cai and Ruizhe Chen and Keqin Chen and Xionghui Chen and Zesen Cheng and Lianghao Deng and Wei Ding and Chang Gao and Chunjiang Ge and Wenbin Ge and Zhifang Guo and Qidong Huang and Jie Huang and Fei Huang and Binyuan Hui and Shutong Jiang and Zhaohai Li and Mingsheng Li and Mei Li and Kaixin Li and Zicheng Lin and Junyang Lin and Xuejing Liu and Jiawei Liu and Chenglong Liu and Yang Liu and Dayiheng Liu and Shixuan Liu and Dunjie Lu and Ruilin Luo and Chenxu Lv and Rui Men and Lingchen Meng and Xuancheng Ren and Xingzhang Ren and Sibo Song and Yuchong Sun and Jun Tang and Jianhong Tu and Jianqiang Wan and Peng Wang and Pengfei Wang and Qiuyue Wang and Yuxuan Wang and Tianbao Xie and Yiheng Xu and Haiyang Xu and Jin Xu and Zhibo Yang and Mingkun Yang and Jianxin Yang and An Yang and Bowen Yu and Fei Zhang and Hang Zhang and Xi Zhang and Bo Zheng and Humen Zhong and Jingren Zhou and Fan Zhou and Jing Zhou and Yuanzhi Zhu and Ke Zhu},
      journal={arXiv preprint arXiv:2511.21631},
      year={2025}
}
```

## Methods

Training methods Bulwark implements (BibTeX from arXiv):

- LoRA adapters on the encoder:

  ```bibtex
  @misc{hu2021loralowrankadaptationlarge,
        title={LoRA: Low-Rank Adaptation of Large Language Models},
        author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
        year={2021},
        eprint={2106.09685},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2106.09685},
  }
  ```

- Uncertainty-weighted multi-task loss:

  ```bibtex
  @misc{kendall2018multitasklearningusinguncertainty,
        title={Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics},
        author={Alex Kendall and Yarin Gal and Roberto Cipolla},
        year={2018},
        eprint={1705.07115},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/1705.07115},
  }
  ```

- Soft-target distillation from B1 to E1:

  ```bibtex
  @misc{hinton2015distillingknowledgeneuralnetwork,
        title={Distilling the Knowledge in a Neural Network},
        author={Geoffrey Hinton and Oriol Vinyals and Jeff Dean},
        year={2015},
        eprint={1503.02531},
        archivePrefix={arXiv},
        primaryClass={stat.ML},
        url={https://arxiv.org/abs/1503.02531},
  }
  ```
