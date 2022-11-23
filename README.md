# HwinArion

A speech recognition package for controlling your computer.


## Installation

### Required Non-Python Dependencies

Install [PortAudio 19](http://www.portaudio.com/).
You may need to install the development libraries for PortAudio.

Follow the instructions for [installing Pydub dependencies](https://github.com/jiaaro/pydub#dependencies).

### Optional Dependencies

#### Pocketsphinx

To install for Pocketsphinx, you will need to follow their [installation instructions](https://github.com/bambocher/pocketsphinx-python#installation) for installing the non-Python dependencies.

#### Whisper

Whisper depends on PyTorch, but adding pytorch as a dependency through poetry is difficult (see [PyTorch issue](https://github.com/pytorch/pytorch/issues/26340), [Poetry issue](https://github.com/python-poetry/poetry/issues/4231)).
While PyTorch is listed as a dependency for whisper, you may need to manually install PyTorch by following their installation instructions: https://pytorch.org/get-started/locally/

### Installing HwinArion

Once you have installed the non-Python dependencies and any optional dependencies, install HwinArion with `poetry install`.
