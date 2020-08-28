# Protein Embedding Search

**This is not an officially supported Google product.**

Protein database search tools such as BLAST are instrumental for research in the life sciences. However, they are slow and are based on surface-level sequence similarity. We are exploring using neural networks to improve the speed and accuracy of finding relevant sequences from these databases. 

More specifically, we are aiming to learn fixed-length protein embeddings using [contextual lenses](https://arxiv.org/pdf/2002.08866.pdf). Generally speaking, a sequence level protein representation, such as a one-hot encoding, is an array of the the form (sequence_length, n) where n is the amino acid embedding dimension. A contextual lens is a (learnable) map from the (sequence_length, n)-array to an (m,)-vector where m is independent of sequence_length. Embeddings are constructed using an encoder function followed by a contextual lens. To learn these embeddings a downstream prediction task is performed using a single dense layer. Gradients are backpropagated through all 3 components of the architecture (encoder, lens, predictor) using the Adam optimizer with variable (potentially zero) learning rates per component.

### Encoders
- [One-hot](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/contextual_lenses/encoders.py#L21): non-learnable
- [CNN](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/contextual_lenses/encoders.py#L46): learnable
- [Transformer](https://github.com/google-research/google-research/blob/master/protein_lm/models.py#L870): learnable and pretrainable

### Lenses
- [Mean/Max-Pool](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/contextual_lenses/contextual_lenses.py#L21): non-learnable
- [Linear-Mean/Max-Pool](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/contextual_lenses/contextual_lenses.py#L46): learnable
- [GatedConvolution](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/contextual_lenses/contextual_lenses.py#L125): learnable and self-attentive

## TAPE Protein Engineering Tasks
[TAPE](https://arxiv.org/pdf/1906.08230.pdf) proposes two protein engineering tasks: fluorescence prediction and stability prediction. We implement our lens architectures on these tasks in a [Google Colab notebook](https://github.com/amirshane/protein-embedding-retrieval/blob/master/tape_contextual_lenses.ipynb). We find that for the fluorescence task both linear regression on the one-hot encodings and 1-layer convolution compete with and outperform the best pretrained language models in TAPE. Likewise, we find that for the stability task 3-layer convolution competes with and outperforms TAPE's models. See below for a table of our results compared to TAPE's results.

### Fluorescence
|             |       | Full Test Set | Bright Mode Only | Dark Mode Only |
| ----------- | ----- | ------------- | ---------------- | -------------- |
| **Model Type**  | Model | rho | mse | rho | mse        | rho | mse      |
| ----------- | ----- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

### Stability
| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

## Downstream Task
The downstream task we use to train embeddings is Pfam family classification. We pick an encoder and a lens and train the architecture to predict a protein's family using only its primary sequence. We train on 10000 families in the data set. We then take the embeddings from this trained model and use them to do family prediction on 1000 holdout families with KNN (using 1 neighbor). This test allows us to assess the extent of transfer learning by seeing how much the embeddings have learned about the holdout families from the train families. In theory, a perfect model would map all proteins that are members of the same family to a single vector. To test for this we run our KNN classification with 1 sample (where the KNN classifier only sees 1 protein per family), 5 samples, 10 samples, and 50 samples. 

### Pretraining
We also measure the effect that pretraining has on the performance of a language model encoder. There has been a great deal of interest in measuring the degree to which pretraining protein language models improves their performance on downstream tasks. TAPE investigates this and proposes baselines. Our results indicate that pretraining offers a substantial boost in performance on the family classification task. We use transformer language models, specifically BERT models similar to to the the [ProGen model](https://www.biorxiv.org/content/10.1101/2020.03.07.982272v2.full.pdf) and the [models used by FAIR](https://www.biorxiv.org/content/10.1101/622803v2.full.pdf). Our [models](https://github.com/google-research/google-research/tree/master/protein_lm) are implemented in jax/flax and pretrained on the [TrEMBL protein corpus](https://www.uniprot.org/statistics/TrEMBL).

## Quickstart
The first step is to install [Caliban](https://github.com/google/caliban). We use Caliban for running individual jobs and parallelizing many jobs on GCP (Google Cloud Platform).

For a simple demo on your machine (recommended only if it is equipped with a GPU) run
```
caliban run --experiment_config demo.json pfam_experiment.py
```

To run on cloud first connect to a GCP project (equipped with GPU resources) by running
```
gcloud init
```
If your GCP project is named MY_PROJECT run
```
export PROJECT_ID=MY_PROJECT
```
Finally, run
```
caliban cloud --experiment_config demo.json pfam_experiment.py
```
The demo takes ~3 hours with an Nvidia Tesla P100 GPU (the Caliban default)

## Reproducing our Results
To reproduce our results connect to a GCP project, ideally with a large number of GPUs, and run
```
caliban cloud --experiment_config params_combinations.json pfam_experiment.py
```

## Source Code Headers

Every file containing source code must include copyright and license
information. This includes any JS/CSS files that you might be serving out to
browsers. (This is to help well-intentioned people avoid accidental copying that
doesn't comply with the license.)

Apache header:

    Copyright 2020 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
