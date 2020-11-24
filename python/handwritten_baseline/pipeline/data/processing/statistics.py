from typing import Dict
from tabulate import tabulate

from python import DOCUMENT_ID, TOPIC_ID, SUBTOPIC, TOKEN, SENTENCE_IDX, EVENT, MENTION_ID, TOKEN_IDX_FROM, TOKEN_IDX_TO
from python.handwritten_baseline import MENTION_TYPE_COARSE
from python.handwritten_baseline.pipeline.data.base import Dataset, BaselineDataProcessorStage


class StatisticsStage(BaselineDataProcessorStage):

    def __init__(self, pos, config, config_global, logger):
        super(StatisticsStage, self).__init__(pos, config, config_global, logger)

        self.print_examples = config.get("print_examples", False)

    def _process_dataset(self,
                         dataset: Dataset,
                         live_objects: Dict) -> Dataset:
        num_topics = len(dataset.documents.index.get_level_values(TOPIC_ID).unique())
        num_subtopics = len(dataset.documents.index.to_frame()[[TOPIC_ID, SUBTOPIC]].drop_duplicates())
        num_documents = len(dataset.documents)

        avg_subtopics_per_topic = num_subtopics / num_topics
        avg_documents_per_subtopic = num_documents / num_subtopics
        avg_documents_per_topic = num_documents / num_topics

        num_tokens = len(dataset.tokens)
        num_types = len(dataset.tokens[TOKEN].unique())

        num_sentences = len(dataset.tokens.index.to_frame()[[DOCUMENT_ID, SENTENCE_IDX]].drop_duplicates())

        num_action_mentions = len(dataset.mentions_action)
        num_action_mentions_stacked = dataset.mentions_action.reset_index().duplicated([DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX_FROM, TOKEN_IDX_TO], keep=False).sum()
        num_action_clusters = len(dataset.mentions_action[EVENT].unique())
        num_action_singleton_clusters = (dataset.mentions_action[EVENT].value_counts() == 1).sum()

        num_participant_mentions = len(dataset.mentions_participants) if dataset.mentions_participants is not None else 0
        num_location_mentions = len(dataset.mentions_location) if dataset.mentions_location is not None else 0
        num_time_mentions = len(dataset.mentions_time) if dataset.mentions_time is not None else 0

        clusters_by_size = dataset.mentions_action[EVENT].value_counts()
        cluster_size_distribution = clusters_by_size.value_counts().sort_index()
        cluster_size_distribution.name = "num-occurrences"
        cluster_size_distribution.index.name = "cluster-size"

        # how many mentions have participants, time, location linked
        if dataset.semantic_roles is not None:
            # count number of participant, time and location mention for each action mention
            num_mentions_of_coarse_type_per_mention = dataset.semantic_roles.groupby([DOCUMENT_ID, MENTION_ID])[MENTION_TYPE_COARSE].value_counts()
            num_mentions_of_coarse_type_per_mention.name = "num-mentions"
            # pivot so that we have [doc-id, mention-id] as the index and the number of location/participant/time mentions for each mention as the columns
            num_mentions_of_coarse_type_per_mention = num_mentions_of_coarse_type_per_mention.reset_index().pivot([DOCUMENT_ID, MENTION_ID], MENTION_TYPE_COARSE, "num-mentions")

            # number of mentions which have at least one argument (location/participant/time) linked, regardless of type
            num_mentions_with_linked_args = len(num_mentions_of_coarse_type_per_mention)

            # number of mentions which have at least one argument (location/participant/time) linked, by type - absolute and relative percentage
            num_mentions_with_args_by_type = {type_: len(num_mentions_of_coarse_type_per_mention.loc[num_mentions_of_coarse_type_per_mention[type_] > 0]) for type_ in ["location", "participants", "time"]}
            num_mentions_with_args_by_type_relative = {type_: num / num_mentions_with_linked_args for type_, num in num_mentions_with_args_by_type.items()}
        else:
            num_mentions_with_linked_args = 0
            num_mentions_with_args_by_type = {}
            num_mentions_with_args_by_type_relative = {}

        # NOTE: we can get the number of coreference link by type (within-doc, within-subtopic, cross-subtopic, cross-topic from the mention pair generator during training), so we don't need to repeat that here

        with (self.stage_disk_location / "statistics.txt").open("w") as f:
            f.write(f"""
Number of topics: {num_topics}
Number of subtopics: {num_subtopics}
Number of documents: {num_documents}
Avg. subtopics per topic: {avg_subtopics_per_topic}
Avg. documents per subtopic: {avg_documents_per_subtopic}
Avg. documents per topic: {avg_documents_per_topic}
Number of tokens: {num_tokens}
Number of types: {num_types}
Number of sentences: {num_sentences}
Number of participant mentions: {num_participant_mentions}
Number of time mentions: {num_time_mentions}
Number of location mentions: {num_location_mentions}
Number of action mentions: {num_action_mentions}
Number of stacked action mentions (identical span): {num_action_mentions_stacked}
Number of event clusters: {num_action_clusters}
Number of singleton event clusters: {num_action_singleton_clusters}

Largest clusters:
{tabulate(clusters_by_size.head(10).to_frame("num-mentions"), headers="keys")}

Cluster size distribution:
{tabulate(cluster_size_distribution.to_frame(), headers="keys")}

""")

            if dataset.semantic_roles is not None:
                f.write(f"""
Number of action mentions with linked arguments: {num_mentions_with_linked_args}
Number of action mentions with linked arguments of type:
{num_mentions_with_args_by_type}

Number of action mentions with linked arguments of type (relative):
{num_mentions_with_args_by_type_relative}
""")


        # print some examples if that's desired
        if self.print_examples:
            # print sentences with stacked annotations
            mentions_action = dataset.mentions_action.reset_index()
            mentions_action_stacked = mentions_action.loc[mentions_action.duplicated([DOCUMENT_ID, SENTENCE_IDX, TOKEN_IDX_FROM, TOKEN_IDX_TO], keep=False)]
            sentences_with_stacked_actions = mentions_action_stacked[[DOCUMENT_ID, SENTENCE_IDX]].drop_duplicates()
            for _, row in sentences_with_stacked_actions.iterrows():
                sentence = " ".join(dataset.tokens.loc[tuple(row), TOKEN])
                print(f"{row.values} - {sentence}")

            # for each event cluster, collect all sentences referencing that event
            sentences_with_event_mentions = dataset.mentions_action.reset_index()[[DOCUMENT_ID, SENTENCE_IDX]].drop_duplicates()
            sentences_with_event_mentions = sentences_with_event_mentions.merge(dataset.tokens.reset_index(), on=[DOCUMENT_ID, SENTENCE_IDX])
            sentences_with_event_mentions = sentences_with_event_mentions.groupby([DOCUMENT_ID, SENTENCE_IDX])[TOKEN].apply(list)
            mentions_action_with_sentences = dataset.mentions_action.reset_index().merge(sentences_with_event_mentions, on=[DOCUMENT_ID, SENTENCE_IDX])
            mentions_action_with_sentences.to_json(self.stage_disk_location / "event_mentions_with_sentences.json", orient="records")

            # print full documents with event actions highlighted
            for idx, mention in dataset.mentions_action.iterrows():
                doc_id, mention_id = idx
                sent_idx = mention[SENTENCE_IDX]
                token_idx_from = mention[TOKEN_IDX_FROM]
                token_idx_to = mention[TOKEN_IDX_TO]
                dataset.tokens.at[(doc_id, sent_idx, token_idx_from), TOKEN] = ">>>" + dataset.tokens.at[(doc_id, sent_idx, token_idx_from), TOKEN]
                dataset.tokens.at[(doc_id, sent_idx, token_idx_to-1), TOKEN] = dataset.tokens.at[(doc_id, sent_idx, token_idx_to-1), TOKEN] + "<<<"

            sentences = dataset.tokens[TOKEN].groupby([DOCUMENT_ID, SENTENCE_IDX]).apply(lambda l: " ".join(l.values))
            documents = sentences.groupby(DOCUMENT_ID).apply(lambda l: "\n".join(l.values))
            documents.to_csv(self.stage_disk_location / "documents_with_event_actions_marked.csv")
        return dataset

component = StatisticsStage
