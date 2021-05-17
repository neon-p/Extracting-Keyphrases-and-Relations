
import os
import pandas as pd
import numpy as np
from nltk.tokenize import TreebankWordTokenizer


def load_data_with_char_offsets(data_folder):
    """
    Load the training, dev or test set for ScienceIE. The text will not be
    tokenized to allow processing with methods that use their own tokenizers.
    The entity annotations are expressed as character offsets, that is, as
    indexes into the string of characters. Each entity within a document has an
    ID. The relations refer to the entities using these IDs.

    Args:
        data_folder -- the path to the original ScienceIE dataset for either the train, dev or test split.

    Returns:
        docs -- a list, where each entry corresponds to a document. The list
        entry for each document is a tuple, (text, entities, relations),
        where text is a string, entities is a list of entity annotations and
        relations is a list of relation annotations. Each entity is a tuple,
        (ID, type, start_char_offset, end_char_offset). Each relation is a tuple,
        (type, entity1_ID, entity2_ID).

    """
    flist = os.listdir(data_folder)
    fileids = []
    docs = []
    all_rels = []

    for f in flist:
        if not f.endswith(".ann"):
            continue
        fileids.append(f)

        # print(f'processing file {f}')
        f_text = open(os.path.join(data_folder, f.replace(".ann", ".txt")), "rU")

        # there's only one line, as each .ann file is one text paragraph
        for l in f_text:
            text = l

        # a list to store (text, entities, relations) tuples
        doc = [text, [], []]

        # load the annotation data
        doc_annos = pd.read_csv(
            os.path.join(data_folder, f), sep='\t',
            skiprows=0, index_col=0, names=['annottion', 'span']
        )

        for i, anno in enumerate(doc_annos.values):
            tokens = anno[0].split(' ')
            label = tokens[0]
            if '-of' in label:  # relation
                # split this into binary relations -- synonym lists can have multiple
                for eidx, ent1 in enumerate(tokens[1:]):
                    for ent2 in tokens[1+eidx:]:
                        if ent1 == ent2:
                            continue

                        if ':' in ent1:
                            ent1 = ent1.split(':')[1]
                        if ':' in ent2:
                            ent2 = ent2.split(':')[1]
                        doc[2].append((label, ent1, ent2))

            else:  # entity
                entity_id = doc_annos.index[i]
                if len(tokens) > 3:
                    label, start, _, end = tokens
                else:
                    label, start, end = tokens

                start = int(start)
                end = int(end)
                doc[1].append((entity_id, start, end))

        docs.append(doc)

    return docs


def load_tokenized_data(data_folder):
    """
    Load the training, dev or test set for ScienceIE in tokenized format as
    a set of documents as [[(token, label), ...], ...]

    Args:
        data_folder -- the path to the original ScienceIE dataset for either the train, dev or test split.

    Returns:
        docs -- a list, where each entry is a document.
        The list entry for each document is a list of tuples, (token, label).
        The labels are BIO tags, wher 'B' and 'I' tags also include the type of entity.

        all_rels -- a list, where each entry corresponds to a document.
        The entry for each document is a list of tuples,
        (label, entity1_start_token_index, entity2_start_token_index).
        Therefore the relations are referred to by the index of their first tokens.
    """
    flist = os.listdir(data_folder)
    fileids = []
    docs = []
    all_rels = []

    tokenizer = TreebankWordTokenizer()

    for f in flist:
        if not f.endswith(".ann"):
            continue
        fileids.append(f)

#         print(f'processing file {f}')
        f_text = open(os.path.join(data_folder, f.replace(".ann", ".txt")), "rU")

        # there's only one line, as each .ann file is one text paragraph
        for l in f_text:
            text = l

        # a list to store (word, tag) pairs
        doc = []

        # load the annotation data
        doc_annos = pd.read_csv(
            os.path.join(data_folder, f), sep='\t', skiprows=0, index_col=0,
            names=['annottion', 'span']
        )

        tok_idxs = list(tokenizer.span_tokenize(text))

        # empty list that we put annotations into
        entity_toks = {}
        rels = []

        for i, anno in enumerate(doc_annos.values):
            tokens = anno[0].split(' ')
            label = tokens[0]
            if '-of' in label:  # relation
                if len(tokens) > 3:  # split this into binary relations -- synonym lists can have multiple
                    for eidx, ent1 in enumerate(tokens[1:]):
                        for ent2 in tokens[1+eidx:]:
                            if ent1 == ent2:
                                continue

                            if ':' in ent1:
                                ent1 = ent1.split(':')[1]
                            if ':' in ent2:
                                ent2 = ent2.split(':')[1]

                            rels.append([label, ent1, ent2])
                            # print(f'{label}, {ent1}, {ent2}, {tokens}')
                else:
                    label, ent1, ent2 = tokens

                if ':' in ent1:
                    ent1 = ent1.split(':')[1]
                if ':' in ent2:
                    ent2 = ent2.split(':')[1]

                rels.append([label, ent1, ent2])
                # print(f'{label}, {ent1}, {ent2}')

            else:  # entity
                if len(tokens) > 3:
                    label, start, _, end = tokens
                else:
                    label, start, end = tokens

                start = int(start)
                end = int(end)

                start_tok, end_tok = get_token_idx_of_spans(tok_idxs, start, end)
                entity_id = doc_annos.index[i]
                entity_toks[entity_id] = (start_tok, end_tok, label)
                # print(f'{entity_id}, {start_tok}, {end_tok}, {label}')

        # put tokens into a list and label all as O for outside to start with
        for tok_span in tok_idxs:
            tok_start = tok_span[0]
            tok_end = tok_span[1]

            token = text[tok_start:tok_end]
            doc.append((token, 'O'))

        # put the entity annotations into the sequence of tokens by changing the O labels within spans
        for entity_label in entity_toks:
            start_tok = entity_toks[entity_label][0]
            end_tok = entity_toks[entity_label][1]
            label = entity_toks[entity_label][2]
            doc[start_tok] = (doc[start_tok][0], 'B-' + label)
            for t in range(start_tok+1, end_tok+1):
                doc[t] = (doc[t][0], 'I-' + label)

        # convert the relation ids to token ids
        # print(rels)
        for i in range(len(rels)):
            rels[i][1] = entity_toks[rels[i][1]][0]
            rels[i][2] = entity_toks[rels[i][2]][0]

        all_rels.append(rels)
        docs.append(doc)

    return docs, all_rels, fileids



def get_token_idx_of_spans(token_chars, span_start, span_end):
    token_chars = np.array(token_chars, dtype=[('start', int), ('end', int)])
    start_chars = token_chars['start']
    end_chars = token_chars['end']

    if span_start in start_chars:
        start_tok = np.argwhere(start_chars == span_start).flatten()[0]
    else:
#         print('no matching start')
        for i, val in enumerate(start_chars):
            if val < span_start:
                lower = val
                start_tok = i
            else:
                break

    if span_end in end_chars:
        end_tok = np.argwhere(end_chars == span_end).flatten()[0]
    else:
#         print('no matching end')
        for i, val in enumerate(end_chars):
            if val >= span_end:
                upper = val
                end_tok = i
                break
#         print(f'replaced {span_end} with {upper}')
#         print(np.max(end_chars))
    return start_tok, end_tok
