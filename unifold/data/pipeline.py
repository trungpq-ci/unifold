# Copyright 2021 Beijing DP Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the Uni-Fold model."""

import os
from pathlib import Path
from typing import Mapping, Optional, Sequence
from absl import logging
from unifold.common import residue_constants
from unifold.data import parsers
from unifold.data import templates
from unifold.data.tools import hhblits
from unifold.data.tools import hhsearch
from unifold.data.tools import jackhmmer
import numpy as np
import time

# Internal import (7716).

FeatureDict = Mapping[str, np.ndarray]


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

  num_res = len(msas[0][0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  return features


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               pdb70_database_path: str,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000):
    """Constructs a feature dict for a given FASTA file."""
    #   "debug
    self.uniref90_database_path = uniref90_database_path
    self.jackhmmer_binary_path = jackhmmer_binary_path
    # end

    self._use_small_bfd = use_small_bfd
    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path,
        get_tblout=True)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path)
    else:
      self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path])
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path)
    self.hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path,
        databases=[pdb70_database_path])
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    t_0 = time.time()
    jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
        input_fasta_path)[0]
    logging.info("Search: Uniref90 JackHHMER time: {} sec".format(time.time() - t_0))
    t_0 = time.time()
#    jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(
#        input_fasta_path)[0]
#    logging.info("Search: MGnify JackHHMER time: {} sec".format(time.time() - t_0))
#
#    t_0 = time.time()
#    uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
#        jackhmmer_uniref90_result['sto'], max_sequences=self.uniref_max_hits)
#    hhsearch_result = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)
#    logging.info("Search: pdb70 hhsearch time: {} sec".format(time.time() - t_0))

    t_0 = time.time()
    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    with open(uniref90_out_path, 'w') as f:
      f.write(jackhmmer_uniref90_result['sto'])

    # debug
    t_0 = time.time()
    hits_path = Path("./hits").joinpath(Path(msa_output_dir).parts[-2], 'uniref90_hits.name')
    hits_path.parent.mkdir(exist_ok=True, parents=True)
    names = []
    with open(hits_path, 'w') as f:
      for line in str(jackhmmer_uniref90_result['sto']).splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '//')):
            continue
        name, sequence = line.split()
        name = name.split("/")[0]
        if name not in names:
            names.append(name)
            f.write(name+"\n")
    logging.info("Uniref90 same name time: {} sec".format(time.time() - t_0))


#    mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
#    with open(mgnify_out_path, 'w') as f:
#      f.write(jackhmmer_mgnify_result['sto'])
#
#    pdb70_out_path = os.path.join(msa_output_dir, 'pdb70_hits.hhr')
#    with open(pdb70_out_path, 'w') as f:
#      f.write(hhsearch_result)
#    logging.info("Dumping: uniref90, mgnify, pdb70 time: {} sec".format(time.time() - t_0))
#
#    uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(
#        jackhmmer_uniref90_result['sto'])
#    mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(
#        jackhmmer_mgnify_result['sto'])
#    hhsearch_hits = parsers.parse_hhr(hhsearch_result)
#    mgnify_msa = mgnify_msa[:self.mgnify_max_hits]
#    mgnify_deletion_matrix = mgnify_deletion_matrix[:self.mgnify_max_hits]
#
#    if self._use_small_bfd:
#      t_0 = time.time()
#      jackhmmer_small_bfd_result = self.jackhmmer_small_bfd_runner.query(
#          input_fasta_path)[0]
#      logging.info("Search: small bfd time: {} sec".format(time.time() - t_0))
#
#      t_0 = time.time()
#      bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.a3m')
#      with open(bfd_out_path, 'w') as f:
#        f.write(jackhmmer_small_bfd_result['sto'])
#      logging.info("Dumping: small bfd time: {} sec".format(time.time() - t_0))
#
#      bfd_msa, bfd_deletion_matrix, _ = parsers.parse_stockholm(
#          jackhmmer_small_bfd_result['sto'])
#    else:
#      hhblits_bfd_uniclust_result = self.hhblits_bfd_uniclust_runner.query(
#          input_fasta_path)
#
#      bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
#      with open(bfd_out_path, 'w') as f:
#        f.write(hhblits_bfd_uniclust_result['a3m'])
#
#      bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(
#          hhblits_bfd_uniclust_result['a3m'])
#
#    t_0 = time.time()
#    templates_result = self.template_featurizer.get_templates(
#        query_sequence=input_sequence,
#        query_pdb_code=None,
#        query_release_date=None,
#        hits=hhsearch_hits)
#
#    sequence_features = make_sequence_features(
#        sequence=input_sequence,
#        description=input_description,
#        num_res=num_res)
#
#    msa_features = make_msa_features(
#        msas=(uniref90_msa, bfd_msa, mgnify_msa),
#        deletion_matrices=(uniref90_deletion_matrix,
#                           bfd_deletion_matrix,
#                           mgnify_deletion_matrix))
#    logging.info("Make MSA feature time: {} sec".format(time.time() - t_0))
#
#    logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
#    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
#    logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
#    logging.info('Final (deduplicated) MSA size: %d sequences.',
#                 msa_features['num_alignments'][0])
#    logging.info('Total number of templates (NB: this can include bad '
#                 'templates and is later filtered to top 4): %d.',
#                 templates_result.features['template_domain_names'].shape[0])
#
#    return {**sequence_features, **msa_features, **templates_result.features}

  def process_from_hits(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    # create sub-dataset from uniref90
    t_0 = time.time()
    hits_path = Path("./hits").joinpath(Path(msa_output_dir).parts[-2], 'uniref90_hits.name')
    sub_uniref90_path = "temp_uniref90_subset.fasta"
    with open(hits_path, "r") as f, \
         open(self.uniref90_database_path, "r") as g, \
         open(sub_uniref90_path, "w") as h:
        for name in f:
            try:
                while True:
                    data_line = next(g)
                    if not data_line or not data_line.startswith(('>')):
                        continue
                    data_name = data_line.split()[0][1:]
                    if name == data_name:
                        h.write(data_line)
                        h.write(g.readline())
                        h.flush()
                        break
            except StopIteration:
                break
        h.close()
    logging.info("Subset database: Uniref90 : {} sec".format(time.time() - t_0))

    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=self.jackhmmer_binary_path,
        database_path=sub_uniref90_path,
        get_tblout=True)

    t_0 = time.time()
    jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
        input_fasta_path)[0]
    logging.info("Search: Uniref90 JackHHMER time: {} sec".format(time.time() - t_0))

    t_0 = time.time()
    uniref90_out_path = os.path.join(str(msa_output_dir), 'uniref90_hits.sto')
    with open(uniref90_out_path, 'w') as f:
      f.write(jackhmmer_uniref90_result['sto'])

    return

