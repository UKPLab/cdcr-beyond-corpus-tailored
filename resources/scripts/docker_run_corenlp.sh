#!/usr/bin/env bash
java -Xmx16G -cp "/corenlp/stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 600000 -maxCharLength 500000 -quiet True -preload ssplit,tokenize,pos,lemma,depparse,parse