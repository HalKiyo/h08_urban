#!/bin/bash
# to convert original filenames to temporal filenames
# 2024/04/06 kajiyama TokyoTech

#--------------------------------------------------------------
# Setting (Edit here to city SUFFIX)
#--------------------------------------------------------------

original_extentions=".bk5"

#--------------------------------------------------------------
# Input (Edit here according to your H08 directory path)
#--------------------------------------------------------------

RGNDIR='/mnt/c/Users/tsimk/Downloads/dotfiles/h08/region/dat'

#--------------------------------------------------------------
# Output (Edit here according to your load directory path)
#--------------------------------------------------------------

H08DIR='/mnt/c/Users/tsimk/Downloads/H08_20230612'

#--------------------------------------------------------------
# Job (restore name)
#--------------------------------------------------------------

cd "${RGNDIR}"

METLST="/LWdown__ /PSurf___ /Rainf___ /SWdown__ /Wind____ /Prcp____ /Qair____ /Snowf___ /Tair____"
OUTLST="/riv_ara_ /riv_mou_ /riv_num_ /riv_nxd_ /riv_nxl_ /riv_seq_"

find "$RGNDIR" -type f -name "*${original_extentions}" -print0 | while IFS= read -r -d $'\0' file; do
    found=false
    for item in $METLST; do
        if [[ $file == *"$item"* ]]; then
            new_file=$(echo "$file" | sed 's|dotfiles/h08/region|H08_20230612/met|')
            cp $file $new_file
            found=true
            break
        fi
    done
    if [ "$found" = false ]; then
        for item in $OUTLST; do
            if [[ $file == *"$item"* ]]; then
                new_file=$(echo "$file" | sed 's|dotfiles/h08/region/dat|H08_20230612/map/out|')
                cp $file $new_file
                echo "$new_file"
                found=true
                break
            fi
        done
    fi
    if [ "$found" = false ]; then
        new_file=$(echo "$file" | sed 's|dotfiles/h08/region|H08_20230612/map|')
        cp $file $new_file
        echo "$new_file"
    fi
done
