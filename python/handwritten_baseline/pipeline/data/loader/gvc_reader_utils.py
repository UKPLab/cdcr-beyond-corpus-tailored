import datetime
from typing import Optional, Tuple

import pandas as pd

from python import DOCUMENT_ID, PUBLISH_DATE, SUBTOPIC, MENTION_ID, SENTENCE_IDX, TOKEN_IDX_FROM, TOKEN_IDX_TO, EVENT, \
    SENTENCE_TYPE, TOKEN_IDX, TOKEN, EVENT_ID, TOPIC_ID
from python.util.ftfy import clean_string


def load_gvc_dataset(path: str, doc_to_subtopic_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contents_rows = []
    documents_rows = []
    mentions_rows = []

    with open(path, 'r') as file:
        doc_id = None
        dct = None
        iter_sentence_index = -1
        last_sentence_index = None
        iter_mention_id = 0
        token_idx_from = None
        token_idx_offset = 0    # we need this to correct indices of tokens following "NEWLINE" tokens, which we remove

        for line in file:
            if line.startswith("#begin"):
                doc_id = line.strip().split()[2][1:-2]
            elif line.startswith("#end"):
                doc_row = {DOCUMENT_ID: doc_id, PUBLISH_DATE: dct}
                documents_rows.append(doc_row)

                # reset variables for next document
                doc_id = None
                dct = None
                iter_sentence_index = -1
                last_sentence_index = None
                iter_mention_id = 0
                token_idx_from = None
                token_idx_offset = 0
            else:
                parts = line.strip().split("\t")
                token_idx_conflated, token, sentence_type, label_text = parts

                # detect special lines with publish date information
                if sentence_type == "DCT":
                    dct = datetime.datetime.strptime(token, "%Y-%m-%d")
                    continue

                # there are over 5000 useless NEWLINE tokens in the corpus, skip those
                if token == "NEWLINE":
                    token_idx_offset -= 1
                    continue

                # there are tokens with unicode garbage in the corpus, for example some \x92 in
                # 254c63ca82173008f14f769c20db88e0: remove those, and if the token is empty after removal, skip it
                token = clean_string(token).strip()
                if not token:
                    token_idx_offset -= 1
                    continue

                # take apart the token index, the pattern is: 40b69cf630792394ef837aee6c959ece.t1.2
                _, sentence_idx_and_sentence_type, token_idx = token_idx_conflated.split(".")

                # In the original files, there can be a title sentence with sentence index 1 and a body sentence
                # with sentence index 1. We normalize this and number the first sentence 0, the second 1 etc.
                # irrespective of the sentence type.
                if last_sentence_index is None or not sentence_idx_and_sentence_type == last_sentence_index:
                    last_sentence_index = sentence_idx_and_sentence_type
                    iter_sentence_index += 1
                    token_idx_offset = 0

                sentence_idx = iter_sentence_index
                token_idx = int(token_idx) + token_idx_offset

                # For some reason, the token numbering of the first title sentence and the first body sentence of each article starts with 1. We set this manually to 0 again...
                if sentence_idx_and_sentence_type in ["t1", "b1"]:
                    token_idx -= 1

                if not label_text == "-":
                    label = int(label_text.replace("(", "").replace(")", ""))
                    if "(" in label_text:
                        token_idx_from = token_idx
                    if ")" in label_text:
                        token_idx_to = token_idx + 1
                        mentions_rows.append(
                            {DOCUMENT_ID: doc_id, MENTION_ID: iter_mention_id, SENTENCE_IDX: sentence_idx,
                             TOKEN_IDX_FROM: token_idx_from, TOKEN_IDX_TO: token_idx_to, EVENT: label})
                        iter_mention_id += 1
                content_line = {DOCUMENT_ID: doc_id, SENTENCE_TYPE: sentence_type, SENTENCE_IDX: sentence_idx,
                                TOKEN_IDX: token_idx, TOKEN: token}
                contents_rows.append(content_line)
    contents = pd.DataFrame(contents_rows)
    contents.set_index([DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX], inplace=True)
    contents.sort_index(inplace=True)

    if not doc_to_subtopic_file:
        raise ValueError
    doc_to_subtopic = pd.read_csv(doc_to_subtopic_file, index_col=0)

    documents = pd.DataFrame(documents_rows)
    documents[SUBTOPIC] = documents[DOCUMENT_ID].map(doc_to_subtopic[EVENT_ID]).astype(str)
    documents[TOPIC_ID] = "gun_violence"

    documents.set_index([TOPIC_ID, SUBTOPIC, DOCUMENT_ID], inplace=True)
    documents.sort_index(inplace=True)
    documents[DOCUMENT_ID] = documents.index.get_level_values(DOCUMENT_ID)  # add doc-id back as a data column

    mentions = pd.DataFrame(mentions_rows)
    mentions.set_index([DOCUMENT_ID, MENTION_ID], inplace=True)
    mentions.sort_index(inplace=True)

    return documents, contents, mentions