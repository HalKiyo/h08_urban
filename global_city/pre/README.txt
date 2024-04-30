-------------------------------------------------------------------

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

-------------------------------------------------------------------

# citycenter
make_citycenter_gpw3.sh => gpw3
make_citycenter_gpw4.sh => gpw4
make_citycenter_gpw4_30s.py => gpw4(30seconds)

# citymask
make_citymask_kajiyama.py => original(full WUP)
make_downtown.py => gpw4 + downtown(/dat/cty_msk_/gpw4/)=> pickle(/dat/dwn_twn_)
make_cluster.py => gpw4 + downtown(city_cluster)

# jogesui
explore_josui_gesui.py => original
explore_jogesui_ctycls.py => gpw4 + downtown(city_cluster)

# intake
explore_intake.py => original
explore_int_ctycls.py => gpw4 + downtown(city_cluster)
