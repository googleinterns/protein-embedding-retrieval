# Protein Embedding Search

**This is not an officially supported Google product.**

Protein database search tools such as [BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi) are instrumental for research in the life sciences. However, they are slow and are based on surface-level sequence similarity. We are exploring using neural networks to improve the speed and accuracy of finding relevant sequences from these databases. 

More specifically, we are aiming to learn fixed-length protein embeddings using [contextual lenses](https://arxiv.org/pdf/2002.08866.pdf). Generally speaking, a sequence level protein representation, such as a one-hot encoding, is an array of the the form (sequence_length, n) where n is the amino acid embedding dimension. A contextual lens is a (learnable) map from the (sequence_length, n)-array to an (m,)-vector where m is independent of sequence_length. Embeddings are constructed using an encoder function followed by a contextual lens. To learn these embeddings a downstream prediction task is performed using a single dense layer. Gradients are backpropagated through all 3 components of the architecture (encoder, lens, predictor) using the Adam optimizer with variable (potentially zero) learning rates and weight decays per component.

### Encoders
- [One-hot](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/contextual_lenses/encoders.py#L21): non-learnable
- [CNN](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/contextual_lenses/encoders.py#L46): learnable
- [Transformer](https://github.com/google-research/google-research/blob/master/protein_lm/models.py#L870): learnable and pretrainable

### Lenses
- [Mean/Max-Pool](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/contextual_lenses/contextual_lenses.py#L21): non-learnable
- [Linear-Mean/Max-Pool](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/contextual_lenses/contextual_lenses.py#L46): learnable
- [GatedConvolution](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/contextual_lenses/contextual_lenses.py#L125): learnable and self-attentive

## TAPE Protein Engineering Tasks
[TAPE](https://arxiv.org/pdf/1906.08230.pdf) proposes two protein engineering tasks: fluorescence prediction and stability prediction. We implement our lens architectures on these tasks in a [Google Colab notebook](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/cnn_protein_landscapes.ipynb). We find that for the fluorescence task both linear regression on the one-hot encodings and 1-layer convolution compete with and outperform the best pretrained language models in TAPE. Likewise, we find that for the stability task 3-layer convolution competes with and outperforms TAPE's models. See below for our results compared to TAPE's results (bold represents best performance).

### Fluorescence
MSE is mean squared error and rho represents Spearman's rank correlation coefficient.
| Model Type | Model | Full Test Set (MSE, rho) | Bright Mode (MSE, rho) | Dark Mode (MSE, rho) |
| ---------- | ----- | :----------------------: | :--------------------: | :------------------: |
| Baseline | Linear Regression | (0.35, **0.69**) | (0.088, **0.68**) | (0.329, **0.05**) |
| Lens Architecture  | 1-Layer CNN + MaxPool | (0.26, **0.69**) | (0.09, 0.65) | (0.26, **0.05**) |
| Lens Architecture  | 1-Layer CNN + LinearMaxPool | (0.23, **0.69**) | (0.12, 0.66) | (0.28, **0.05**) |
| TAPE | Best of all models | (**0.19**, 0.68) | (**0.07**, 0.63) | (**0.22**, **0.05**)|


### Stability
Accuracy (Acc) is measured using the parent protein as a decision boundary and labeling mutations as beneficial if predicted stability is greater than predicted parent stability and deleterious if the opposite is true. rho represents Spearman's rank correlation coefficient. The letters A and B represent the alpha and beta topologies, respectively.
| Model Type | Model | Full Test Set (rho, Acc) | AAA (rho, Acc) | ABBA (rho, Acc) | BABB (rho, Acc) | BBABB (rho, Acc) |
| ---------- | ----- | :---------------------------: | :-----------------: | :------------------: | :------------------: | :-------------------: |
| Baseline | Linear Regression | (0.49, 0.60) | (0.21, 0.66) | (-0.03, 0.6) | (0.51, 0.64) | (0.38, 0.61) |
| Lens Architecture  | 3-Layer CNN + MaxPool | (0.76, 0.75) | (0.69, **0.71**) | (0.37, 0.70) | (0.50, 0.72) | (0.60, 0.68) |
| Lens Architecture  | Dilated 3-Layer CNN + MaxPool | (0.75, 0.73) | (0.67, 0.69) | (0.49, 0.69) | (0.61, 0.70) | (0.53, 0.64) |
| Lens Architecture  | 3-Layer CNN + LinearMaxPool | (0.71, **0.77**) | (0.59, 0.69) | (**0.52**, 0.77) | (0.55, **0.73**) | (0.60, **0.70**) |
| Lens Architecture  | Ensemble (Average) of above CNN models | (**0.79**, **0.77**) | (0.67, **0.71**) | (**0.53**, 0.75) | (0.65, **0.74**) | (0.60, **0.70**) |
| TAPE | Best of all models | (0.73, 0.70) | (**0.72**, 0.70) | (0.48, **0.79**) | (**0.68**, 0.71) | (**0.67**, **0.70**) |


## Downstream Task
The downstream task we use to train embeddings is Pfam family classification. We pick an encoder and a lens and train the architecture to predict a protein's family using only its primary sequence. We train on 10000 families in the data set and measure **Lens Accuracy**: the accuracy achieved on the *test set of train families* by the architecture trained for family prediction on the *train set of train families*. We then take the embeddings from this trained model and use them to do family prediction on 1000 holdout families with KNN (using 1 neighbor). This test allows us to assess the extent of transfer learning by seeing how much the embeddings have learned about the holdout families from the train families. In theory, a perfect model would map all proteins that are members of the same family to a single vector. To test for this we measure **n-Sample Test KNN Accuracy**: The accuracy achieved on the *test set of test families* by a KNN classifier trained on the embeddings (from our architecture) of the *train set of test families* using *at most n samples per family*.

### Pretraining
We also measure the effect that pretraining has on the performance of a language model encoder. There has been a great deal of interest in measuring the degree to which pretraining protein language models improves their performance on downstream tasks. TAPE investigates this and proposes baselines. Our results indicate that pretraining offers a substantial boost in performance on the family classification task. We use transformer language models, specifically BERT models similar to to the the [ProGen model](https://www.biorxiv.org/content/10.1101/2020.03.07.982272v2.full.pdf) and the [models used by FAIR](https://www.biorxiv.org/content/10.1101/622803v2.full.pdf). Our [models](https://github.com/google-research/google-research/tree/master/protein_lm) are implemented in jax/flax and pretrained on the [TrEMBL protein corpus](https://www.uniprot.org/statistics/TrEMBL).

## Results
In the table below we show the accuracies achieved using KNN on the model embeddings as well as KNN using BLAST's weighted edit distance. We show a simple 2-layer CNN, 3 different size language models both with and without pretraining, and the [Blundell CNN model](https://www.biorxiv.org/content/10.1101/626507v4.full.pdf). All bolded numbers represent better performance compared to BLAST. The key takeways are the performance of pretrained language models on 1-sample classification and the substantial performance boost from pretraining said models. 

| Model                                      | 1-Sample Accuracy | 5-Sample Accuracy | 10-Sample Accuracy | 50-Sample Accuracy |
|--------------------------------------------|:-----------------:|:-----------------:|:------------------:|:------------------:|
| BLAST                                      | 0.860750          | 0.978355          | 0.991342           | 0.996392           |
| 2-layer CNN                                | 0.687815          | 0.870944          | 0.914924           | 0.956741           |
| Small Transformer                          | 0.769286          | 0.920692          | 0.952415           | 0.974045           |
| Pretrained Small Transformer               | **0.873828**      | 0.968998          | 0.979813           | 0.992790           |
| Medium Transformer                         | 0.778659          | 0.921413          | 0.956741           | 0.981255           |
| Pretrained Medium Transformer              | **0.863775**      | 0.968277          | 0.984859           | 0.994232           |
| Large Transformer                          | 0.749820          | 0.894737          | 0.937996           | 0.970440           |
| Pretrained Large Transformer               | **0.865898**      | 0.974045          | 0.984859           | 0.995674           |
| Blundell Lens-Family CNN                   | **0.877345**      | **0.980519**      | **0.992063**       | 0.993506           |
| Blundell Full-Family CNN**                 | **0.923521**      | **0.984848**      | **0.992785**       | 0.995671           |
| Blundell Full-Family CNN** w/ Whitening*** | **0.940837**      | **0.988456**      | **0.996392**       | **0.996392**       |

** The Full-Family Blundell CNN is not performing transfer learning. It was trained on families that appear in the KNN task.

*** Whitened embeddings are obtained by performing PCA on the embeddings of all Pfam seed sequences and applying the corresponding whitening transformation to the KNN train and test sequences.

Below we show plots of the top 10 n-Sample Test KNN Accuracies vs. Lens Accuracies for different models and for n = 1, 5, 10, 50. The key takeaways are the noticable boost pretraining the language models provides, the fact that Lens Accuracy is not a perfect predictor of Test KNN Accuracy, and the independence of performance and transformer size.

![1-sample](/figures/1-sample_test_knn_accuracy.png)

![5-sample](/figures/5-sample_test_knn_accuracy.png)

![10-sample](/figures/10-sample_test_knn_accuracy.png)

![50-sample](/figures/50-sample_test_knn_accuracy.png)

## Quickstart
To clone this project run
```
git clone --recurse-submodules https://github.com/googleinterns/protein-embedding-retrieval.git
```

Once the project is cloned, the first step is to install [Caliban](https://github.com/google/caliban). We use Caliban for running individual jobs and parallelizing many jobs on GCP (Google Cloud Platform).

For a simple demo on your machine (recommended only if it is equipped with a GPU) run
```
caliban run --experiment_config demo.json pfam_experiment.py
```
The demo takes ~3 hours with an Nvidia Tesla P100 GPU (the Caliban default). By default this will load data from and save data to the 'neuralblast_public' GCS bucket. You can change this by modifying the values of 'load_gcs_bucket' and 'save_gcs_bucket' in demo.json.

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


## Reproducing our Results
To reproduce our results you first need to connect to a GCP project, ideally one with a large number of GPUs, and clone the project. Then take the [generate_params.py](https://github.com/googleinterns/protein-embedding-retrieval/blob/master/generate_params.py) script, and modify the variable 'save_gcs_bucket' in the call to main() using the GCS bucket you wish to save to (and potentially do the same for the one you want to load from 'load_gcs_bucket'). Run the script to generate the appropriate parameter combinations and run
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
