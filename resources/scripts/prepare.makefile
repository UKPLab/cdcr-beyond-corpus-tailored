RESOURCES=resources

# use bash in order to be able to use `source ...` to activate venvs
SHELL := /bin/bash



########################### Dataset download / setup ###########################
GVC_URL='https://github.com/cltl/GunViolenceCorpus/raw/9ecca9ef3a7083e36f0445d9ed3e3bc5b2e80393/gold.conll'
ECBP_URL='http://kyoto.let.vu.nl/repo/ECB+_LREC2014.zip'
FCCT_URL='https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2305'

ECBP_DESTINATION=$(RESOURCES)/data/ecbplus
GVC_DESTINATION=$(RESOURCES)/data/gun_violence
FCCT_DESTINATION=$(RESOURCES)/data/football

.PHONY: datasets clean_datasets ecbplus gvc fcct

datasets: | ecbplus gvc fcct

clean_datasets:
	rm -rf $(ECBP_DESTINATION)
	rm -rf $(FCCT_DESTINATION)
	rm -f $(GVC_DESTINATION)/GVC_gold.conll

# download ECB+
$(ECBP_DESTINATION):
	mkdir -p $@

$(ECBP_DESTINATION)/ECB+_LREC2014.zip: | $(ECBP_DESTINATION)
	cd $(ECBP_DESTINATION); wget $(ECBP_URL)

ecbplus: $(ECBP_DESTINATION)/ECB+_LREC2014.zip
	unzip -q $< -d $(ECBP_DESTINATION)
	unzip -q $(ECBP_DESTINATION)/ECB+_LREC2014/ECB+.zip -d $(ECBP_DESTINATION)

# create splits
	cd $(ECBP_DESTINATION) && mkdir train valid test guns sports
	cd $(ECBP_DESTINATION)/ECB+ && \
	cp -r 1 3 4 6 7 8 9 10 11 13 14 16 19 20 22 24 25 26 27 28 29 30 31 32 33 ../train/ && \
	cp -r 2 5 12 18 21 23 34 35 ../valid/ && \
	cp -r 36 37 38 39 40 41 42 43 44 45 ../test/ && \
	cp -r 3 8 16 18 22 33 ../guns/ && \
	cp -r 5 7 10 25 29 31 ../sports/

# move over sentence filter csv
	mv $(ECBP_DESTINATION)/ECB+_LREC2014/ECBplus_coreference_sentences.csv $(ECBP_DESTINATION)

# remove some stuff
	rm -r $(ECBP_DESTINATION)/ECB+ $(ECBP_DESTINATION)/__MACOSX

# "download" FCC-T
$(FCCT_DESTINATION):
	mkdir -p $@

fcct: | $(FCCT_DESTINATION)
	@echo ""
	@echo ""
	@echo "#########################################################################################################################"
	@echo "#   Please follow $(FCCT_URL) for generating the FCC-T corpus.        #"
	@echo "#   Then, copy the contents of the 'datasets' directory into the '$(FCCT_DESTINATION)' directory in this project.   #"
	@echo "#########################################################################################################################"
	@echo ""
	@echo ""

# download GVC
$(GVC_DESTINATION):
	mkdir -p $@

$(GVC_DESTINATION)/GVC_gold.conll: | $(GVC_DESTINATION)
	cd $(GVC_DESTINATION); wget $(GVC_URL); mv gold.conll GVC_gold.conll

gvc: $(GVC_DESTINATION)/GVC_gold.conll

################################################################################




############################# Wikidata embeddings  #############################

# see readme of https://github.com/facebookresearch/PyTorch-BigGraph
EMBEDDINGS_URL='https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_vectors.npy.gz'
EMBEDDINGS_INDEX_URL='https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1_names.json.gz'
EMBEDDINGS_DESTINATION=$(RESOURCES)/wikidata_embeddings

.PHONY: embeddings clean_embeddings

clean_embeddings:
	rm -rf $(EMBEDDINGS_DESTINATION)

$(EMBEDDINGS_DESTINATION):
	mkdir -p $@

# download and unpack embeddings file
$(EMBEDDINGS_DESTINATION)/wikidata_translation_v1_vectors.npy.gz: | $(EMBEDDINGS_DESTINATION)
	cd $(EMBEDDINGS_DESTINATION); wget -c $(EMBEDDINGS_URL)

$(EMBEDDINGS_DESTINATION)/wikidata_translation_v1_vectors.npy: $(EMBEDDINGS_DESTINATION)/wikidata_translation_v1_vectors.npy.gz
	gunzip -k $<

# download and unpack index file
$(EMBEDDINGS_DESTINATION)/wikidata_translation_v1_names.json.gz: | $(EMBEDDINGS_DESTINATION)
	cd $(EMBEDDINGS_DESTINATION); wget -c $(EMBEDDINGS_INDEX_URL)

$(EMBEDDINGS_DESTINATION)/wikidata_translation_v1_names.json: $(EMBEDDINGS_DESTINATION)/wikidata_translation_v1_names.json.gz
	gunzip -k $<

embeddings: $(EMBEDDINGS_DESTINATION)/wikidata_translation_v1_vectors.npy $(EMBEDDINGS_DESTINATION)/wikidata_translation_v1_names.json

################################################################################



################################# Virtualenvs  #################################

ALLENNLP_OLD_VENV=venv_allennlp_0.9.0
ALLENNLP_NEW_VENV=venv_allennlp_1.0.0

.PHONY: venvs clean_venvs
venvs: $(ALLENNLP_OLD_VENV) $(ALLENNLP_NEW_VENV)

clean_venvs:
	rm -rf $(ALLENNLP_OLD_VENV)
	rm -rf $(ALLENNLP_NEW_VENV)

$(ALLENNLP_OLD_VENV):
	python3.7 -m venv $(ALLENNLP_OLD_VENV)
	source $(ALLENNLP_OLD_VENV)/bin/activate && \
	pip install --upgrade pip==20.2.2 && \
	pip install wheel && \
	pip install -r resources/requirements/allennlp_0.9.0.txt && \
	deactivate

$(ALLENNLP_NEW_VENV):
	python3.7 -m venv $(ALLENNLP_NEW_VENV)
	source $(ALLENNLP_NEW_VENV)/bin/activate && \
	pip install --upgrade pip==20.2.2 && \
	pip install wheel && \
	pip install -r resources/requirements/allennlp_1.0.0.txt && \
	deactivate

	ln -s $(ALLENNLP_NEW_VENV) venv

################################################################################



################################## SpanBERT  ###################################

# To make this span embedder work for predicting in our AllenNLP model, we need to modify the model checkpoint to look
# exactly like AllenNLP would expect, i.e. we have to remove some irrelevant weights.

SPANBERT_LARGE_URL=https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz
SPANBERT_DESTINATION=$(RESOURCES)/spanbert

.PHONY: spanbert clean_spanbert
spanbert: $(SPANBERT_DESTINATION)/model.tar.gz

clean_spanbert:
	rm -rf $(SPANBERT_DESTINATION)

$(SPANBERT_DESTINATION):
	mkdir -p $@

$(SPANBERT_DESTINATION)/coref-spanbert-large-2020.02.27.tar.gz: | $(SPANBERT_DESTINATION)
	cd $(SPANBERT_DESTINATION); wget -c $(SPANBERT_LARGE_URL)

$(SPANBERT_DESTINATION)/model.tar.gz: $(SPANBERT_DESTINATION)/coref-spanbert-large-2020.02.27.tar.gz $(ALLENNLP_OLD_VENV)
	cd $(SPANBERT_DESTINATION) && tar -xzf coref-spanbert-large-2020.02.27.tar.gz

# modify weights.th and copy over the right config file
	cd $(SPANBERT_DESTINATION); mv weights.th weights_original.th
	source $(ALLENNLP_OLD_VENV)/bin/activate && python3 resources/scripts/modify_spanbert_checkpoint.py $(SPANBERT_DESTINATION)/weights_original.th $(SPANBERT_DESTINATION)/weights.th
	cp resources/scripts/spanbert_config.json $(SPANBERT_DESTINATION)/config.json

# pack new archive and remove unneeded files
	cd $(SPANBERT_DESTINATION) && \
	tar -czf model.tar.gz weights.th config.json vocabulary && \
	rm -rf weights.th config.json vocabulary weights_original.th

################################################################################


################################### CoreNLP ####################################

CORENLP_URL=https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
CORENLP_DESTINATION=$(RESOURCES)/corenlp

.PHONY: corenlp clean_corenlp

clean_corenlp:
	rm -rf $(CORENLP_DESTINATION)

$(CORENLP_DESTINATION):
	mkdir -p $@

$(CORENLP_DESTINATION)/stanford-corenlp-full-2018-10-05.zip: | $(CORENLP_DESTINATION)
	cd $(CORENLP_DESTINATION); wget -c $(CORENLP_URL)

$(CORENLP_DESTINATION)/stanford-corenlp-full-2018-10-05: $(CORENLP_DESTINATION)/stanford-corenlp-full-2018-10-05.zip
	unzip -q $< -d $(CORENLP_DESTINATION)

corenlp: $(CORENLP_DESTINATION)/stanford-corenlp-full-2018-10-05

################################################################################


.DEFAULT_GOAL: all
.PHONY: all clean

all: embeddings venvs spanbert datasets corenlp

clean: clean_embeddings clean_venvs clean_spanbert clean_datasets clean_corenlp