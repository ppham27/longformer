from . import file_utils
from . import dataset_utils
from . import sentencepiece_pb2
import functools
import bisect
import json
import os
import re
import string
from tqdm import tqdm
from absl import logging
import random
import nltk
import numpy as np
import argparse
import sentencepiece as spm
import tensorflow as tf

_CLS_PIECE = '<ans>'
_EOS_PIECE = '</s>'
_SEP_PIECE = '<sep_0>'
_PARAGRAPH_SEP_PIECE = '<sep_1>'
_NULL_PIECE = '<empty>'
_QUESTION_PIECE = '<unused_34>'


def normalize_answer(s, remove_articles=True, keep_punc=None):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def white_space_fix(text):
    return ' '.join(text.split())

  s = s.replace('_', ' ').lower()
  exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
  for punc in (keep_punc or []):
    exclude.remove(punc)
  s = ''.join(ch if ch not in exclude else ' ' for ch in s)
  if remove_articles:
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
  return white_space_fix(s).strip()

def get_text(qad, domain):
  local_file = os.path.join(args.web_dir, qad['Filename']) if domain == 'SearchResults' else os.path.join(args.wikipedia_dir, qad['Filename'])
  return file_utils.get_file_contents(local_file, encoding='utf-8')


def make_inputs(question, text):
  ids = sp_model.EncodeAsIds(question) + [sp_model.PieceToId(_SEP_PIECE)]
  global_ids = [sp_model.PieceToId(_CLS_PIECE)] + [sp_model.PieceToId(_QUESTION_PIECE)] * len(ids)
  segment_ids = [i + 1 for i in range(len(ids))]  # offset for CLS token
  sentences = []
  offsets, offset = [-1] * len(ids), 0
  for paragraph in text.split('\n'):
    paragraph_sentences = [
        s.strip() for s in sent_tokenize.tokenize(paragraph) if len(s.strip()) > 0]
    # if (paragraph_sentences and len(ids) < args.max_num_tokens and
    #     len(global_ids) < args.max_num_global_tokens):
    #   id = sp_model.PieceToId(_PARAGRAPH_SEP_PIECE)
    #   if not (len(ids) > 0 and ids[-1] == id):
    #     ids.append(id)
    #     segment_ids.append(len(global_ids))
    #     if sentences: offset -= 1  # Remove extra sentence padding.
    #     sentences.append(_PARAGRAPH_SEP_PIECE)
    #     offsets.append(offset)
    #     offset += len(_PARAGRAPH_SEP_PIECE)
    for sentence in paragraph_sentences:
      spt = sentencepiece_pb2.SentencePieceText.FromString(
          sp_model.EncodeAsSerializedProto(sentence))
      if (len(ids) + len(spt.pieces) > args.max_num_tokens - 1 or
          len(global_ids) == args.max_num_global_tokens):
        break
      for i, piece in enumerate(spt.pieces):
        ids.append(piece.id)
        segment_ids.append(len(global_ids))
        offsets.append(offset + piece.begin)
        if i == 0 and sentences: offsets[-1] -= 1  # Space after previous sentence.
      offset += len(spt.text.encode('utf-8')) + 1
      # assert spt.pieces[-1].end - spt.pieces[0].begin == len(spt.text.encode('utf-8'))
      sentences.append(spt.text)
      global_ids.append(sp_model.PieceToId(_EOS_PIECE))
  bytes_text = ' '.join(sentences).replace(
      f' {_PARAGRAPH_SEP_PIECE}', _PARAGRAPH_SEP_PIECE).replace(
          f'{_PARAGRAPH_SEP_PIECE} ', _PARAGRAPH_SEP_PIECE).encode('utf-8')
  ids.append(sp_model.PieceToId(_NULL_PIECE))
  offsets.append(len(bytes_text))
  segment_ids.append(0)
    # print(text[:200])
    # print(len(new_text), len(new_text.encode('utf-8')))
  # for i in range(100):
  #   print(bytes_text[offsets[i]:(offsets[i] + 5)], '|' , sp_model.IdToPiece(ids[i]))
  # print(len(ids), len(global_ids))
  # print(sp_model.DecodeIds(global_ids))
  return global_ids, ids, offsets, segment_ids, bytes_text


def add_triple_data(datum, page, domain):
  qad = {'Source': domain}
  for key in ['QuestionId', 'Question', 'Answer']:
    if key == 'Answer' and key not in datum:
      qad[key] = {'NormalizedAliases': []}
      qid = datum['QuestionId']
      print(f'qid: {qid} does not have an answer.')
    else:
      qad[key] = datum[key]
  for key in page:
    qad[key] = page[key]
  return qad


def get_qad_triples(data):
  qad_triples = []
  for datum in data['Data']:
    for key in ['EntityPages', 'SearchResults']:
      for page in datum.get(key, []):
        qad = add_triple_data(datum, page, key)
        qad_triples.append(qad)
  return qad_triples


def _make_int64_feature(xs):
  xs = tf.nest.flatten(xs)
  return tf.train.Feature(int64_list=tf.train.Int64List(value=xs))


def _make_bytes_feature(xs):
  xs = tf.nest.flatten(xs)
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=xs))


def convert_to_examples(qa_json_file):
  qa_json = dataset_utils.read_triviaqa_data(qa_json_file)
  qad_triples = get_qad_triples(qa_json)
  random.seed(args.seed)
  random.shuffle(qad_triples)
  stats = {
      'max_num_answers': 0, 'max_num_global_ids': 0, 'max_num_long_ids': 0,
      'num_triplets_answerless': 0, 'num_triplets': 0}

  data = []
  for qad in tqdm(qad_triples):
    feature = {
        'qid': _make_bytes_feature(qad['QuestionId'].encode('utf-8')),
        'id': _make_bytes_feature(
            dataset_utils.get_question_doc_string(qad['QuestionId'], qad['Filename']).encode('utf-8')),
        'question': _make_bytes_feature(qad['Question'].encode('utf-8')),
    }
    global_token_ids, token_ids, token_offsets, segment_ids, selected_text = make_inputs(
        qad['Question'], get_text(qad, qad['Source']))
    feature['global_token_ids'] = _make_int64_feature(global_token_ids)
    feature['token_ids'] = _make_int64_feature(token_ids)
    feature['segment_ids'] = _make_int64_feature(segment_ids)
    feature['token_offsets'] = _make_int64_feature(token_offsets)
    feature['context'] = _make_bytes_feature(selected_text)
    stats['max_num_long_ids'] = max(stats['max_num_long_ids'], len(token_ids))
    stats['max_num_global_ids'] = max(stats['max_num_global_ids'], len(global_token_ids))
    stats['num_triplets'] += 1
    if qa_json['Split'] == 'test':
      data.append(tf.train.Example(features=tf.train.Features(feature=feature)))
      continue
    answers_in_doc = dataset_utils.answer_index_in_document(
        qad['Answer'], selected_text,
        [
            functools.partial(normalize_answer, remove_articles=False),
            functools.partial(normalize_answer, remove_articles=False, keep_punc=[',', '.']),
            functools.partial(normalize_answer, remove_articles=False, keep_punc=['-']),
            functools.partial(normalize_answer, remove_articles=False, keep_punc=['-', ',', '.']),
        ])
    answer_set = set(qad['Answer']['NormalizedAliases'])
    answers = []
    for answer in answers_in_doc:
      i = bisect.bisect_left(token_offsets, answer['answer_start'])
      if i == len(token_offsets) or answer['answer_start'] < token_offsets[i]: i -= 1
      j = i + 1
      answer_end = answer['answer_start'] + len(answer['text'].encode('utf-8'))
      while j < len(token_offsets) and token_offsets[j] < answer_end: j += 1
      j -= 1
      sp_answer = (
          selected_text[token_offsets[i]:token_offsets[j + 1]] if j + 1 < len(token_offsets)
          else selected_text[token_offsets[i]:])
      if sp_model.IdToPiece(token_ids[i]).startswith('▁') and token_offsets[i] > 0:
        sp_answer = sp_answer[1:]
      sp_answer = normalize_answer(sp_answer.decode('utf-8'))
      if not sp_answer in answer_set:
        # No need to warn if the cause was breaking word boundaries.
        if len(sp_answer) and not len(sp_answer) > len(normalize_answer(answer['text'])):
          logging.warning(
              '%s: "%s" not in %s.', qad['QuestionId'], sp_answer, answer_set)
        continue
      answers.append((i, j))
    answers = set(answers)
    stats['max_num_answers'] = max(stats['max_num_answers'], len(answers))
    if answers:
      answers = [int(a) for a in np.vstack([np.array(answer) for answer in answers]).reshape((-1))]
    else:
      stats['num_triplets_answerless'] += 1
      logging.warning(
          '%s: "%s" has no answer in %s.',
          qad['QuestionId'], qad['Question'], qad['Filename'])
      answers = []
    feature['answers'] = _make_int64_feature(answers)
    data.append(tf.train.Example(features=tf.train.Features(feature=feature)))
    if qa_json['Split'] == 'train' and len(data) >= args.sample_size and qa_json['Domain'] == 'Web':
      break

    if len(data) >= args.sample_size:
      break
  logging.info('Added %d', len(data))
  logging.info(stats)
  return data


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--triviaqa_file', help='Triviaqa file')
  parser.add_argument('--output_file', help='Output TFRecord file.')
  parser.add_argument('--wikipedia_dir', help='Wikipedia doc dir')
  parser.add_argument('--web_dir', help='Web doc dir')

  parser.add_argument('--seed', default=10, type=int, help='Random seed')
  parser.add_argument('--max_num_tokens', default=800, type=int, help='Maximum number of tokens from a document')
  parser.add_argument('--sample_size', default=8000000000000, type=int, help='Random seed')
  parser.add_argument('--tokenizer', default='tokenizers/punkt/english.pickle', help='Sentence tokenizer')
  parser.add_argument('--spm_model', default='vocab_gpt.model', help='SPM model')
  parser.add_argument('--max_num_global_tokens', default=384, type=int, help='Maximum number of global tokens.')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  args = get_args()
  sent_tokenize = nltk.data.load(args.tokenizer)
  sp_model = spm.SentencePieceProcessor()
  with open(args.spm_model, 'rb') as model:
    sp_model.LoadFromSerializedProto(model.read())
  examples = convert_to_examples(args.triviaqa_file)
  with tf.io.TFRecordWriter(
      args.output_file, tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
    for example in examples:
      writer.write(example.SerializeToString())
