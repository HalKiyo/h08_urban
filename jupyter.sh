#!/bin/sh

notebookrun=-notebook
labrun=-lab
if [ $1 = $notebookrun ]; then
    jupyter notebook --allow-root --ip=0.0.0.0 --port=9999 --no-browser
elif [ $1 = $labrun ]; then
    jupyter lab --allow-root --ip=0.0.0.0 --port=9999 --no-browser --NotebookApp.token='' --NotebookApp.password=''
else
    echo "ERROR"
    echo "available_option"
    echo "-notebook"
    echo "-lab"
fi
