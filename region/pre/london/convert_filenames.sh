#!/bin/bash
#######################################################
#to convert original filenames to temporal filenames
#by 2024/02/01, kajiyama TokyoTech
#######################################################
#  Setting (Edit here to city SUFFIX)
#######################################################
original_extension=".ln5"
converted_extension="=ln5"
#######################################################
#  Input (Edit here according to your H08 direcotory path)
###########################################################
H08DIR="/mnt/c/Users/tsimk/Downloads/H08_20230612/"

###########################################################
#  Output (Edit here according to your load direcotory path)
###########################################################
SAVEDIR="/mnt/c/Users/tsimk/Downloads/research/H08/regional_model/fileshare/gdrive20240514/london/"

###########################################################
#  Job (restore name)
###########################################################
cd "${H08DIR}"

find "$H08DIR" -type f -name "*.ln5" -print0 | while IFS= read -r -d $'\0' file; do
    relative_path=$(echo "$file" | sed "s|^$H08DIR||")
    new_file=$(echo "$relative_path" | sed 's/\//-/g')
    save_file=${SAVEDIR}${new_file}
    cp "${H08DIR}${relative_path}" "${save_file//$original_extension/$converted_extension}"
done
