#!/usr/bin/env bash

# install python dependencies
pip install -r requirements.txt

# install spacy module
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz

# sentencepiece==0.1.92 causing segmentation fault, downgrade it to 0.1.91
pip install -I sentencepiece==0.1.91

# download submission models
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


# download submission 1-5 models
for submission_id in 1 #2 3 4 5
do
    rm -rf $PROJECT_DIR/end2end/submission${submission_id}/model
    mkdir $PROJECT_DIR/end2end/submission${submission_id}/model
    wget -P $PROJECT_DIR/end2end/submission${submission_id}/model/ https://storage.googleapis.com/dstc9_submission/submission${submission_id}_model/best_val_model.pth.tar
    wget -P $PROJECT_DIR/end2end/submission${submission_id}/model/ https://storage.googleapis.com/dstc9_submission/submission${submission_id}_model/config.json
    wget -P $PROJECT_DIR/end2end/submission${submission_id}/model/ https://storage.googleapis.com/dstc9_submission/submission${submission_id}_model/merges.txt
    wget -P $PROJECT_DIR/end2end/submission${submission_id}/model/ https://storage.googleapis.com/dstc9_submission/submission${submission_id}_model/vocab.json
done

# set PYTHONPATH
echo "export PYTHONPATH=/root/" > $PROJECT_DIR/.bashrc