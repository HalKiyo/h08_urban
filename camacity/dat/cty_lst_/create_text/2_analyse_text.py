import re
import numpy as np
#---------------------------------------------------------------------------------------------

cama_path = f'/mnt/c/Users/tsimk/Downloads/dotfiles/h08/camacity/dat/cty_lst_/create_text/camacity_second.txt'

data = []
with open(cama_path, 'r') as file:
    for line in file:
        row = line.strip().split('|')
        data.append(row)

vald_list = [int(i[0]) for i in data if i[1] != 'RMVD' and i[2] != 'NoMK']
nomk_list = [int(i[0]) for i in data if i[2] == 'NoMK']
rmvd_list = [int(i[0]) for i in data if i[1] == 'RMVD']
ovlp_list = [int(i[0]) for i in data if i[1] == 'OVLP' and i[2] != 'DOWN']
down_list = [int(i[0]) for i in data if i[1] == 'VALD' and i[2] == 'DOWN']

"""
print(f'vald_list {len(vald_list)}')
print(f'nomk_list {len(nomk_list)}')
print(f'rmvd_list {len(rmvd_list)}')
print(f'ovlp_list {len(ovlp_list)}')
print(f'down_list {len(down_list)}')
"""

#---------------------------------------------------------------------------------------------

full_log = f'/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/pre/log/downtown.log'

with open(full_log, 'r', encoding='utf-8') as file:
    log = file.read()

blocks = log.split('#########################################')

cityindex_222 = []

# 各ブロックごとに処理を行う
for block in blocks:
    # cityindexを見つける
    cityindex_match = re.search(r'cityindex (\d+)', block)
    if cityindex_match:
        cityindex = cityindex_match.group(1)

        # ブロック内に '///222///' が含まれるか確認
        if '/// 222 ///' in block:
            cityindex_222.append(int(float(cityindex)))

# 結果を表示
#print("cityindex with ///222///:", cityindex_222)

#---------------------------------------------------------------------------------------------

#print(vald_list)
#print(cityindex_222)
vald_dupl = list(set(vald_list) & set(cityindex_222))
nomk_dupl = list(set(nomk_list) & set(cityindex_222))
rmvd_dupl = list(set(rmvd_list) & set(cityindex_222))
ovlp_dupl = list(set(ovlp_list) & set(cityindex_222))
down_dupl = list(set(down_list) & set(cityindex_222))
print(len(vald_dupl))
print(len(nomk_dupl))
print(len(rmvd_dupl))
print(len(ovlp_dupl))
print(len(down_dupl))
