LABEL_COREF = 1
LABEL_NON_COREF = 0

# some constants for dataframes
POS = "pos"
LEMMA = "lemma"
MENTION_TEXT = "mention-text"
MENTION_TYPE_COARSE: str = "mention-type-coarse"
COMPONENT_MENTION_ID = "component-mention-id"
TIMEX_NORMALIZED = "timex-normalized"
TIMEX_NORMALIZED_PARSED = "timex-normalized-parsed"

# more constants for dataframes involving entity linking
DBPEDIA_URI = "dbpedia-uri"
LATITUDE = "latitude"
LONGITUDE = "longitude"
WIKIDATA_QID = "wikidata-qid"
GEO_HIERARCHY = "geo-hierarchy"

# additional attributes of the Dataset class, used for feature precomputations etc.
WIKIDATA_EMBEDDINGS = "wikidata_embeddings"
SENTENCE_EMBEDDINGS = "sentence_embeddings"
ACTION_PHRASE_EMBEDDINGS = "action_phrase_embeddings"

# coarse types
NAMED_ENTITY = "named-entity"
TFIDF = "tfidf"
TIME = "temporal"
LOCATION = "spatial"
ACTION = "action"
PARTICIPANTS = "participants"
OTHER = "other"
COARSE_TYPES = [PARTICIPANTS, ACTION, TIME, LOCATION, OTHER]

# feature importance stuff
FEATURE = "feature"
TYPE = "type"
WEIGHT = "weight"

# metrics
RECALL = "R"
PRECISION = "P"
F1 = "F1"
SETTING = "setting"

# prediction output
IDX_A_DOC = "idx-a-doc"
IDX_B_DOC = "idx-b-doc"
IDX_A_MENTION = "idx-a-mention"
IDX_B_MENTION = "idx-b-mention"
PREDICTION = "prediction"
LABEL = "label"
INSTANCE = "instance"

# run-specific stuff
DATASET = "dataset"
MODEL = "model"
JOB_ID = "job-id"
TIMESTAMP = "timestamp"
RANDOM_SEED = "random-seed"