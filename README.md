# Protein Embedding Search

**This is not an officially supported Google product.**

Protein database search tools such as BLAST are instrumental for research in the life sciences. However, they are slow and are based on surface-level sequence similarity. We are exploring using neural networks to improve the speed and accuracy of finding relevant sequences from these databases. More specifically, we are aiming to learn fixed-length protein embeddings using contextual lenses (https://arxiv.org/pdf/2002.08866.pdf). Generally speaking, a sequence level protein representation, such as a one-hot encoding, is an array of the the form (sequence_length, n) where n is the amino acid embedding dimension. A contextual lens is a (learnable) map from the (sequence_length, n)-array to an (m,)-vector where m is independent of sequence_length. Embeddings are constructed using an encoder function followed by a contextual lens. To learn these embeddings a downstream prediction task is performed using a single feedforward layer. Gradients are backpropagated through all 3 components of the architecture (encoder, lens, predictor) using the Adam optimizer with variable (potentially zero) learning rates per component.

## Encoders
- One-hot: non-learnable
- CNN: learnable
- Transformer: learnable and pretrainable

## Lenses
- Mean/Max-Pool: non-learnable
- Linear-Mean/Max-Pool: learnable
- GatedConvolution: learnable and self-attentive

## Downstream Task
The downstream task we use to train embeddings is Pfam family classification. We pick an encoder and a lens and train the architecture to predict a protein's family using only its primary sequence. We train on 10000 families in the data set. We then take the embeddings from this trained model and use them to do family prediction on 1000 holdout families with KNN (using 1 neighbor). This test allows us to assess the extent of transfer learning by seeing how much the embeddings have learned about the holdout families from the train families. In theory, a perfect model would map all proteins that are members of the same family to a single vector. To test for this we run our KNN classification with 1 sample (where the KNN classifier only sees 1 protein per family), 5 samples, 10 samples, and 50 samples. 

### Pretraining
We also measure the effect that pretraining has on the performance of a language model encoder. There has been much interest in measuring the degree to which pretraining protein language models improves their performance on downstream tasks (https://arxiv.org/pdf/1906.08230.pdf). Our results indicate that pretraining offers a substantial boost in performance on the family classification task.

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
