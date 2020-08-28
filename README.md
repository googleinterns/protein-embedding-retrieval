# Protein Embedding Search

**This is not an officially supported Google product.**

Protein database search tools such as BLAST are instrumental for research in the life sciences. However, they are slow and are based on surface-level sequence similarity. We are exploring using neural networks to improve the speed and accuracy of finding relevant sequences from these databases. More specifically, we are aiming to learn fixed-length protein embeddings using contextual lenses (https://arxiv.org/pdf/2002.08866.pdf). Generally speaking, a sequence level protein representation, such as a one-hot encoding, is an array of the the form (sequence_length, n) where n is the amino acid embedding dimension. A contextual lens is a (learnable) map from the (sequence_length, n)-array to an (m,)-vector where m is independent of sequence_length. Embeddings are constructed using an encoder function followed by a contextual lens. To learn these embeddings a downstream prediction task is performed using a single feedforward layer. Gradients are backpropagated through all 3 components of the architecture (encoder, lens, predictor).

### Encoders
- One-hot: non-learnable
- CNN: learnable
- Transformer: learnable and pretrainable

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
