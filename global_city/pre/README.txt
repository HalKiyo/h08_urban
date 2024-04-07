1. make_citycenter.sh
    input:  cty_lst_
    output: cty_cnt_
2. make_citymask_kajiyama.py
    input:  cty_cnt_
            cty_lst_
            lnd_ara_
            pop_tot_
    output: cty_msk_
3. explore_josui_gesui.py
    input:  cty_msk_
            cty_lst_
            riv_num_
            riv_ara_
            riv_nxl_
    output: cty_prf_
            cty_swg_
4. explore_intake.py
    input:  riv_out_
            can_ext_
            elv_min_
            riv_num_
            cty_msk_
            cty_prf_
            cty_cnt_
    output: cty_int_
            cty_int_/fig/
            cty_int_/city_water_intake.txt
