# White-Box LSTMs
Code for the paper [_Evaluating Attribution Methods using White-Box LSTMs_](https://arxiv.org/abs/2010.08606), to appear in the proceedings of [BlackboxNLP](https://blackboxnlp.github.io/).

Dependencies:
* Python 3
* [PyTorch](https://pytorch.org/)
* [Torchtext](https://pytorch.org/text/)
* [Captum](http://www.captum.ai/)

Dependencies for generating heatmaps:
* [Yattag](https://www.yattag.org/)
* [Matplotlib](https://matplotlib.org/)
* [Jupyter Notebook](https://jupyter.org/)

## Usage

Refer to `example.ipynb` to see how to reproduce the heatmaps from the paper. The scripts `lrp_saturation_test.py` and `ablation_test.py` run the LRP saturation test and the ablation test, respectively.

