"""Computes accuracy of 1 nearest neighbor classification using BLAST.

Example usage:
blast_baseline.py \
--train_file=./resources/knn_data/5-samples_train_knn_data_families_15001-16000.csv \
--test_file=./resources/knn_data/test_knn_data_families_15001-16000.csv
"""

import os
import subprocess
import tempfile

from absl import app
from absl import flags
import numpy as np
import pandas as pd

# Need to install locally from https://github.com/google-research/proteinfer
from proteinfer import baseline_utils

# We also assume that blast is installed locally.
# sudo apt-get install ncbi-blast+


flags.DEFINE_string('train_file', '', 'Input train csv file.')
flags.DEFINE_string('test_file', '', 'Input test csv file.')


FLAGS = flags.FLAGS


_BLAST_FLAGS = '-outfmt 6 -max_hsps 1 -num_threads 10 -num_alignments 1'


def _get_header(row):
  accession = row.accession.replace('/', '_').replace('-', '_')
  return '>accession="%s"\tlabels="%s"' % (accession, row.label)


def _get_fasta_entry(row):
  header = _get_header(row)
  return '\n'.join([header, row.sequence])


def _write_fasta(df, output_file):
  entries = df.apply(_get_fasta_entry, axis=1)
  with open(output_file, 'w') as file:
    file.write('\n'.join(entries))


def _run_cmd(cmd_string):
  subprocess.run(cmd_string.split(' '), check=True)


class BlastClassifier(object):
  """Stateful wrapper for BLAST system calls."""

  def __init__(self, df):
    _, self._train_fasta = tempfile.mkstemp()
    _, self._blast_db = tempfile.mkstemp()
    _write_fasta(df, self._train_fasta)
    print(self._train_fasta)
    self._train_df = baseline_utils.load_ground_truth(self._train_fasta)
    self._label_vocab = df.label.unique()
    cmd = 'makeblastdb -in %s -dbtype prot -out %s' % (self._train_fasta,
                                                       self._blast_db)
    _run_cmd(cmd)

  def __del__(self):
    os.remove(self._train_fasta)
    os.remove(self._blast_db)

  def predict(self, df):
    """Predicts labels by propagating labels from BLAST top hit."""

    _, query_fasta = tempfile.mkstemp()
    _, blast_output = tempfile.mkstemp()
    _write_fasta(df, query_fasta)
    cmd = 'blastp -query %s -db %s %s -out %s' % (query_fasta, self._blast_db,
                                                  _BLAST_FLAGS, blast_output)
    _run_cmd(cmd)

    assert df.label.isin(self._label_vocab).all()
    query_df = baseline_utils.load_ground_truth(query_fasta)
    results_df = baseline_utils.load_blast_output(blast_output,
                                                  self._label_vocab,
                                                  self._train_df,
                                                  query_df)

    os.remove(query_fasta)
    os.remove(blast_output)
    return results_df


def _get_label(label_set):
  if label_set:
    assert len(label_set) == 1
    return next(iter(label_set))
  else:
    return None


def _compute_accuracy(df):
  prediction = df.predicted_label.apply(_get_label)
  true_label = df.true_label.apply(_get_label)
  return np.mean(prediction == true_label)


def _load(filename):
  df = pd.read_csv(filename)
  df.rename(columns=dict(sequence_name='accession'), inplace=True)
  return df


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_df = _load(FLAGS.train_file)
  test_df = _load(FLAGS.test_file)

  blast_classifier = BlastClassifier(df=train_df)
  output_df = blast_classifier.predict(test_df)

  accuracy = _compute_accuracy(output_df)
  print('Accuracy = %f' % accuracy)


if __name__ == '__main__':
  app.run(main)
