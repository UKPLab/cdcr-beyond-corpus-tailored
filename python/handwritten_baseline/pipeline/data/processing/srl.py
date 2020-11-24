from typing import Dict
from typing import Tuple, List

import pandas as pd
from allennlp.data.dataset_readers.dataset_utils import bio_tags_to_spans
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

try:
    # allennlp 0.90 import
    from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor
except ImportError as e1:
    try:
        # allennlp 1.0.0rc5 import
        from allennlp_models.structured_prediction.predictors.srl import SemanticRoleLabelerPredictor
    except ImportError as e2:
        raise ValueError("Cannot import SRL predictor!")

from tqdm import tqdm

from python import DOCUMENT_ID
from python import SENTENCE_IDX, TOKEN, TOKEN_IDX_TO, TOKEN_IDX_FROM, MENTION_ID
from python.handwritten_baseline import MENTION_TYPE_COARSE, TIME, LOCATION, PARTICIPANTS, COMPONENT_MENTION_ID
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage
from python.util.spans import span_matching


class SemanticRoleLabelingStage(BaselineDataProcessorStage):

    def __init__(self, pos, config, config_global, logger):
        super(SemanticRoleLabelingStage, self).__init__(pos, config, config_global, logger)

        # see AllenNLP Hub: https://github.com/allenai/allennlp-hub/blob/0838c7b06abb2eb3ef90af193335f32784296694/allennlp_hub/pretrained/allennlp_pretrained.py
        srl_model = config.get("srl_model", "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")

        self._cache = self._provide_cache("srl_cache", bind_parameters=config)

        # this one does not work, some files are missing:
        # elmo_srl_luheng_2018 = "https://allennlp.s3.amazonaws.com/models/srl-model-2020.02.10.tar.gz"

        archive = load_archive(srl_model)
        predictor_name = "semantic-role-labeling"
        self._srl_predictor = Predictor.from_archive(archive, predictor_name)  # type: SemanticRoleLabelerPredictor

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        semantic_roles = []

        # determine sentences with action mentions
        for doc_id, mentions_action_doc in tqdm(dataset.mentions_action.groupby(DOCUMENT_ID),
                                                desc="SRL on documents",
                                                mininterval=10):
            # skip documents for which we already have semantic roles
            if dataset.semantic_roles is not None and doc_id in dataset.semantic_roles[DOCUMENT_ID]:
                continue

            for sent_idx, mentions_action_sent in mentions_action_doc.groupby(SENTENCE_IDX):
                # run SRL:
                # AllenNLP SRL models can accept a tokenized sentence and one verbal predicate and returns argument class probabilities
                # per token, which can be converted to BIO via Viterbi. Notably, there is no possibility to feed in pre-recognized
                # argument spans, so the spans recognized by the model need to be reconciled manually. Also, only _verbal_ predicates
                # are supported.
                tokenized_sentence = dataset.tokens.loc[(doc_id, sent_idx), TOKEN].values

                # predict SRL or obtain from cache
                if tokenized_sentence not in self._cache:
                    srl_prediction = self._srl_predictor.predict_tokenized(tokenized_sentence)
                    self._cache[tokenized_sentence] = srl_prediction
                else:
                    srl_prediction = self._cache[tokenized_sentence]

                # srl_spans: for each verbal predicate in the sentence, a list of tags and their span
                srl_spans = []  # type: List[List[Tuple[str, Tuple[int, int]]]]
                for predicate in srl_prediction["verbs"]:
                    tag_spans_inclusive = bio_tags_to_spans(predicate["tags"])
                    # switch from inclusive span boundaries to exclusive ones
                    tag_spans = [(tag, (start, end + 1)) for (tag, (start, end)) in tag_spans_inclusive]
                    srl_spans.append(tag_spans)

                # (start, end) token indices of each detected verb and preannotated actions in the current sentence
                srl_verb_spans = [(start, end) for predicate_spans in srl_spans for (tag, (start, end)) in
                                  predicate_spans if tag == "V"]
                mention_action_spans = mentions_action_sent[[TOKEN_IDX_FROM, TOKEN_IDX_TO]].values.tolist()

                # Map verbs returned from SRL to action mentions via sentence position: We have n pre-annotated action
                # mentions and m predicates found by SRL. We want to find the best 1:1 assignment from predicate to
                # mention. We approach this as a linear assignment problem.
                map_from_preannotated_action_to_srl_predicate = span_matching(mention_action_spans, srl_verb_spans)

                # for those where mapping exists:
                for i_action, i_predicate in map_from_preannotated_action_to_srl_predicate.items():
                    action = mentions_action_sent.iloc[i_action]
                    action_mention_id = action.name[mentions_action_sent.index.names.index(MENTION_ID)]
                    tag_spans = srl_spans[i_predicate]

                    # map time, location, participants to annotations
                    event_component_rows = []

                    def find_event_component_mapping(mentions_df: pd.DataFrame, srl_target_tags: List[str],
                                                     coarse_mention_type: str):
                        # it can happen that there is no time/location/participant annotated in a sentence; otherwise,
                        # look up mentions in the sentence
                        if not doc_id in mentions_df.index or not sent_idx in mentions_df.loc[doc_id, SENTENCE_IDX]:
                            return
                        mentions_within_doc = mentions_df.loc[doc_id]
                        mentions_within_sentence = mentions_within_doc.loc[mentions_within_doc[SENTENCE_IDX] == sent_idx]

                        mention_spans_within_sentence = mentions_within_sentence[[TOKEN_IDX_FROM, TOKEN_IDX_TO]].values.tolist()
                        _srl_spans = [(start, end) for (tag, (start, end)) in tag_spans if tag in srl_target_tags]
                        mapping = span_matching(mention_spans_within_sentence, _srl_spans)

                        for idx_mention, idx_srl in mapping.items():
                            # 'name' is the only remaining index column here, which is MENTION_ID
                            mapped_mention_id = mentions_within_sentence.iloc[idx_mention].name
                            row = {MENTION_TYPE_COARSE: coarse_mention_type,
                                   COMPONENT_MENTION_ID: mapped_mention_id}
                            event_component_rows.append(row)

                    find_event_component_mapping(dataset.mentions_location, ["ARGM-DIR", "ARGM-LOC"], LOCATION)
                    find_event_component_mapping(dataset.mentions_time, ["ARGM-TMP"], TIME)
                    find_event_component_mapping(dataset.mentions_participants, ["ARG0", "ARG1"], PARTICIPANTS)

                    # Collect it all in a dataframe:
                    # For each action mention:
                    #   - index-y (not an actual index): doc-id, mention-id (this is the action mention id), sent_idx <<-- redundant
                    #   - columns: mention-type-coarse, component-mention-id (the mention associated with its action, and the mention type)
                    if event_component_rows:
                        event_components = pd.DataFrame(event_component_rows)
                        event_components[DOCUMENT_ID] = doc_id
                        event_components[MENTION_ID] = action_mention_id

                        semantic_roles.append(event_components)

        if len(semantic_roles) == 0:
            raise ValueError("No semantic roles found. Possible reasons: (1) Dataset already has semantic roles defined. (2) Pretrained SRL predictor likely does not match allennlp version! Check project README for details.")

        # merge identified event components of each sentence and mention into one dataframe
        semantic_roles = pd.concat(semantic_roles, sort=True)

        # concatenate with existing roles
        if dataset.semantic_roles is not None:
            semantic_roles = pd.concat([semantic_roles, dataset.semantic_roles], ignore_index=True)

        dataset.semantic_roles = semantic_roles
        return dataset


component = SemanticRoleLabelingStage