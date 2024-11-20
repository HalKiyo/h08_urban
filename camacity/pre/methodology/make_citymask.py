import os
import numpy as np

vlddir = f'/mnt/c/Users/tsimk/Downloads/dotfiles/h08/global_city/dat/vld_cty_'
moddir = f'/mnt/c/Users/tsimk/Downloads/dotfiles/h08/camacity/dat/dwn_msk_'
lowdir = f'/mnt/c/Users/tsimk/Downloads/dotfiles/h08/camacity/dat/low_msk_'

savedir = f'/mnt/c/Users/tsimk/Downloads/dotfiles/h08/camacity/dat'
canvas_path = f'{savedir}/cty_msk_/city_clrd0000.gl5'

canvas = np.zeros((2160, 4320))

for cty_ind in range(1, 1861):

###################################################################
# MASK selection
# 1. vld_cty_
# 2. dwn_msk_
# 3. low_msk_
###################################################################

    vld_path = f'{vlddir}/city_{cty_ind:08}.gl5'
    dwn_path = f'{moddir}/city_kj_{cty_ind:08}.gl5'

    if not os.path.exists(vld_path):
        low_path = f'{lowdir}/city_kj_{cty_ind:08}.gl5'

        if not os.path.exists(low_path):
            print(f"{cty_ind} is NoMASK")
            continue

        else:
            msk_path =  low_path

    elif os.path.exists(dwn_path):
        msk_path = dwn_path

    else:
        msk_path = vld_path

###################################################################
# JOB
###################################################################

    # maskの値を取得
    mask = np.fromfile(msk_path, dtype='float32').reshape(2160, 4320)

    # maskが0でない場所のインデックスを取得(tuple形式)
    non_zero_coords = np.where(mask != 0)

    # キャンバスのmask位置に存在する値を取得
    settled_values = canvas[non_zero_coords]

    # ユニークな値は?
    unq = np.unique(settled_values)
    unq_non_zero = unq[unq > 0]

###################################################################
# check overlap
###################################################################

    # overlap判定
    if np.sum(settled_values) > 0:
        # 重複削除
        for j in range(len(mask)):
            for k in range(len(mask[0])):
                if mask[j,k] != 0:
                    if canvas[j,k] != 0:
                        mask[j,k] = 0
        non_zero_update = np.where(mask != 0)

        # update canvas
        canvas[non_zero_update] = cty_ind

        # remove or modify result
        grid_num = np.sum(mask)
        if grid_num == 0:
            print(f'{cty_ind} is removed')
        else:
            print(f'{cty_ind} is modified')

    else:
        # update canvas
        canvas[non_zero_coords] = cty_ind

###################################################################
# save file
###################################################################

canvas.astype(np.float32).tofile(canvas_path)
print(f"{canvas_path} saved")
