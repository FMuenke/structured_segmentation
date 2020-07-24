# Structured Segmentation

## Setup
Installation is done by cloning the repository
```bash
git clone https://github.com/FMuenke/structured_segmentation.git
```
All dependency can be installed with
````bash
pip install -r requirements.txt
````
It is strongly recommended to use a virtual environment (like anaconda).

## Example

ToDo

## Documentation

In this repository there are multiple layer implemented.
The Layer can be stacked and used as one end-to-end trainable model.

Learning Layer:
* GraphLayer
* SuperPixelLayer
* ShapeRefinementLayer

Helper Layer:
* VotingLayer
* BottleNeckLayer
* NormalizationLayer
* InputLayer

In addition this repository provides three pre configured layer configurations (models)

Models:
* EncoderDecoder
* PyramidBoosting
* RandomStructuredRandomForrest
* PatchWork