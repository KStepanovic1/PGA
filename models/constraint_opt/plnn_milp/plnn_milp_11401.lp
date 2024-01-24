\ Model plnn_milp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  8.1817 h[0] + 8.1817 h[1] + 8.1817 h[2] + 8.1817 h[3] + 8.1817 h[4]
   + 8.1817 h[5] + 8.1817 h[6] + 8.1817 h[7] + 8.1817 h[8] + 8.1817 h[9]
   + 8.1817 h[10] + 8.1817 h[11] + 0 p[0] + 0 p[1] + 0 p[2] + 0 p[3]
   + 0 p[4] + 0 p[5] + 0 p[6] + 0 p[7] + 0 p[8] + 0 p[9] + 0 p[10]
   + 0 p[11] + 0 a_time_0_layer_0[0] + 0 a_time_1_layer_0[0]
   + 0 a_time_2_layer_0[0] + 0 a_time_3_layer_0[0] + 0 a_time_4_layer_0[0]
   + 0 a_time_5_layer_0[0] + 0 a_time_6_layer_0[0] + 0 a_time_7_layer_0[0]
   + 0 a_time_8_layer_0[0] + 0 a_time_9_layer_0[0] + 0 a_time_10_layer_0[0]
   + 0 a_time_11_layer_0[0] + 0 a_time_0_layer_0[0] + 0 a_time_1_layer_0[0]
   + 0 a_time_2_layer_0[0] + 0 a_time_3_layer_0[0] + 0 a_time_4_layer_0[0]
   + 0 a_time_5_layer_0[0] + 0 a_time_6_layer_0[0] + 0 a_time_7_layer_0[0]
   + 0 a_time_8_layer_0[0] + 0 a_time_9_layer_0[0] + 0 a_time_10_layer_0[0]
   + 0 a_time_11_layer_0[0]
Subject To
 produced_heat_normalize[0]: - 0.0149655791679138 h[0] + h_nor[0]
   = -0.0739299610894942
 produced_heat_normalize[1]: - 0.0149655791679138 h[1] + h_nor[1]
   = -0.0739299610894942
 produced_heat_normalize[2]: - 0.0149655791679138 h[2] + h_nor[2]
   = -0.0739299610894942
 produced_heat_normalize[3]: - 0.0149655791679138 h[3] + h_nor[3]
   = -0.0739299610894942
 produced_heat_normalize[4]: - 0.0149655791679138 h[4] + h_nor[4]
   = -0.0739299610894942
 produced_heat_normalize[5]: - 0.0149655791679138 h[5] + h_nor[5]
   = -0.0739299610894942
 produced_heat_normalize[6]: - 0.0149655791679138 h[6] + h_nor[6]
   = -0.0739299610894942
 produced_heat_normalize[7]: - 0.0149655791679138 h[7] + h_nor[7]
   = -0.0739299610894942
 produced_heat_normalize[8]: - 0.0149655791679138 h[8] + h_nor[8]
   = -0.0739299610894942
 produced_heat_normalize[9]: - 0.0149655791679138 h[9] + h_nor[9]
   = -0.0739299610894942
 produced_heat_normalize[10]: - 0.0149655791679138 h[10] + h_nor[10]
   = -0.0739299610894942
 produced_heat_normalize[11]: - 0.0149655791679138 h[11] + h_nor[11]
   = -0.0739299610894942
 delivered_heat_L[0]: y[0] <= 0.2638680138932743
 delivered_heat_L[1]: y[1] <= 0.2419229554783707
 delivered_heat_L[2]: y[2] <= 0.2223460688348595
 delivered_heat_L[3]: y[3] <= 0.2054531101989264
 delivered_heat_L[4]: y[4] <= 0.2002431323018629
 delivered_heat_L[5]: y[5] <= 0.1985064730028418
 delivered_heat_L[6]: y[6] <= 0.2011904010104199
 delivered_heat_L[7]: y[7] <= 0.2071897694979476
 delivered_heat_L[8]: y[8] <= 0.2226618250710451
 delivered_heat_L[9]: y[9] <= 0.2333975371013577
 delivered_heat_L[10]: y[10] <= 0.2417650773602778
 delivered_heat_L[11]: y[11] <= 0.2401862961793495
 delivered_heat_G[0]: y[0] >= 0.2438680138932743
 delivered_heat_G[1]: y[1] >= 0.2219229554783707
 delivered_heat_G[2]: y[2] >= 0.2023460688348595
 delivered_heat_G[3]: y[3] >= 0.1854531101989264
 delivered_heat_G[4]: y[4] >= 0.1802431323018629
 delivered_heat_G[5]: y[5] >= 0.1785064730028418
 delivered_heat_G[6]: y[6] >= 0.1811904010104199
 delivered_heat_G[7]: y[7] >= 0.1871897694979476
 delivered_heat_G[8]: y[8] >= 0.2026618250710451
 delivered_heat_G[9]: y[9] >= 0.2133975371013577
 delivered_heat_G[10]: y[10] >= 0.2217650773602778
 delivered_heat_G[11]: y[11] >= 0.2201862961793495
 state_time_step_0_layer_0[0]: - 0.4283530116081238 h_nor[0]
   + z_time_0_layer_0[0] - s_time_0_layer_0[0] = -0.030575066518441
 state_time_step_0_output_layer[0]: tau_in[0]
   - 0.9604629874229431 z_time_0_layer_0[0] = -0.0368359982967377
 state_time_step_0_output_layer[1]: tau_out[0]
   - 0.9853600263595581 z_time_0_layer_0[0] = -0.024433
 state_time_step_0_output_layer[2]: m[0]
   + 0.3904860019683838 z_time_0_layer_0[0] = 0.4039709866046906
 state_time_step_1_layer_0[0]: - 0.6405069828033447 h_nor[0]
   - 0.4283530116081238 h_nor[1] - 0.4036439955234528 tau_in[0]
   - 0.5536800026893616 tau_out[0] + 0.1175940036773682 m[0]
   + z_time_1_layer_0[0] - s_time_1_layer_0[0] = -0.1364639824725203
 state_time_step_1_output_layer[0]: tau_in[1]
   - 0.9604629874229431 z_time_1_layer_0[0] = -0.0368359982967377
 state_time_step_1_output_layer[1]: tau_out[1]
   - 0.9853600263595581 z_time_1_layer_0[0] = -0.024433
 state_time_step_1_output_layer[2]: m[1]
   + 0.3904860019683838 z_time_1_layer_0[0] = 0.4039709866046906
 state_time_step_2_layer_0[0]: - 0.255266010761261 h_nor[0]
   - 0.6405069828033447 h_nor[1] - 0.4283530116081238 h_nor[2]
   - 0.4036439955234528 tau_in[1] - 0.5536800026893616 tau_out[1]
   + 0.1175940036773682 m[1] + z_time_2_layer_0[0] - s_time_2_layer_0[0]
   = -0.1691242970454745
 state_time_step_2_output_layer[0]: tau_in[2]
   - 0.9604629874229431 z_time_2_layer_0[0] = -0.0368359982967377
 state_time_step_2_output_layer[1]: tau_out[2]
   - 0.9853600263595581 z_time_2_layer_0[0] = -0.024433
 state_time_step_2_output_layer[2]: m[2]
   + 0.3904860019683838 z_time_2_layer_0[0] = 0.4039709866046906
 state_time_step_3_layer_0[0]: - 0.0432949997484684 h_nor[0]
   - 0.255266010761261 h_nor[1] - 0.6405069828033447 h_nor[2]
   - 0.4283530116081238 h_nor[3] - 0.4036439955234528 tau_in[2]
   - 0.5536800026893616 tau_out[2] + 0.1175940036773682 m[2]
   + z_time_3_layer_0[0] - s_time_3_layer_0[0] = -0.1554449542644856
 state_time_step_3_output_layer[0]: tau_in[3]
   - 0.9604629874229431 z_time_3_layer_0[0] = -0.0368359982967377
 state_time_step_3_output_layer[1]: tau_out[3]
   - 0.9853600263595581 z_time_3_layer_0[0] = -0.024433
 state_time_step_3_output_layer[2]: m[3]
   + 0.3904860019683838 z_time_3_layer_0[0] = 0.4039709866046906
 state_time_step_4_layer_0[0]: - 0.00444999989122152 h_nor[0]
   - 0.0432949997484684 h_nor[1] - 0.255266010761261 h_nor[2]
   - 0.6405069828033447 h_nor[3] - 0.4283530116081238 h_nor[4]
   - 0.4036439955234528 tau_in[3] - 0.5536800026893616 tau_out[3]
   + 0.1175940036773682 m[3] + z_time_4_layer_0[0] - s_time_4_layer_0[0]
   = -0.1389544523908545
 state_time_step_4_output_layer[0]: tau_in[4]
   - 0.9604629874229431 z_time_4_layer_0[0] = -0.0368359982967377
 state_time_step_4_output_layer[1]: tau_out[4]
   - 0.9853600263595581 z_time_4_layer_0[0] = -0.024433
 state_time_step_4_output_layer[2]: m[4]
   + 0.3904860019683838 z_time_4_layer_0[0] = 0.4039709866046906
 state_time_step_5_layer_0[0]: 0.0122410003095865 h_nor[0]
   - 0.00444999989122152 h_nor[1] - 0.0432949997484684 h_nor[2]
   - 0.255266010761261 h_nor[3] - 0.6405069828033447 h_nor[4]
   - 0.4283530116081238 h_nor[5] - 0.4036439955234528 tau_in[4]
   - 0.5536800026893616 tau_out[4] + 0.1175940036773682 m[4]
   + z_time_5_layer_0[0] - s_time_5_layer_0[0] = -0.1295081317155054
 state_time_step_5_output_layer[0]: tau_in[5]
   - 0.9604629874229431 z_time_5_layer_0[0] = -0.0368359982967377
 state_time_step_5_output_layer[1]: tau_out[5]
   - 0.9853600263595581 z_time_5_layer_0[0] = -0.024433
 state_time_step_5_output_layer[2]: m[5]
   + 0.3904860019683838 z_time_5_layer_0[0] = 0.4039709866046906
 state_time_step_6_layer_0[0]: 0.005097 h_nor[0]
   + 0.0122410003095865 h_nor[1] - 0.00444999989122152 h_nor[2]
   - 0.0432949997484684 h_nor[3] - 0.255266010761261 h_nor[4]
   - 0.6405069828033447 h_nor[5] - 0.4283530116081238 h_nor[6]
   - 0.4036439955234528 tau_in[5] - 0.5536800026893616 tau_out[5]
   + 0.1175940036773682 m[5] + z_time_6_layer_0[0] - s_time_6_layer_0[0]
   = -0.1294895382990438
 state_time_step_6_output_layer[0]: tau_in[6]
   - 0.9604629874229431 z_time_6_layer_0[0] = -0.0368359982967377
 state_time_step_6_output_layer[1]: tau_out[6]
   - 0.9853600263595581 z_time_6_layer_0[0] = -0.024433
 state_time_step_6_output_layer[2]: m[6]
   + 0.3904860019683838 z_time_6_layer_0[0] = 0.4039709866046906
 state_time_step_7_layer_0[0]: - 0.0112509997561574 h_nor[0]
   + 0.005097 h_nor[1] + 0.0122410003095865 h_nor[2]
   - 0.00444999989122152 h_nor[3] - 0.0432949997484684 h_nor[4]
   - 0.255266010761261 h_nor[5] - 0.6405069828033447 h_nor[6]
   - 0.4283530116081238 h_nor[7] - 0.4036439955234528 tau_in[6]
   - 0.5536800026893616 tau_out[6] + 0.1175940036773682 m[6]
   + z_time_7_layer_0[0] - s_time_7_layer_0[0] = -0.1365506938949472
 state_time_step_7_output_layer[0]: tau_in[7]
   - 0.9604629874229431 z_time_7_layer_0[0] = -0.0368359982967377
 state_time_step_7_output_layer[1]: tau_out[7]
   - 0.9853600263595581 z_time_7_layer_0[0] = -0.024433
 state_time_step_7_output_layer[2]: m[7]
   + 0.3904860019683838 z_time_7_layer_0[0] = 0.4039709866046906
 state_time_step_8_layer_0[0]: - 0.000658 h_nor[0]
   - 0.0112509997561574 h_nor[1] + 0.005097 h_nor[2]
   + 0.0122410003095865 h_nor[3] - 0.00444999989122152 h_nor[4]
   - 0.0432949997484684 h_nor[5] - 0.255266010761261 h_nor[6]
   - 0.6405069828033447 h_nor[7] - 0.4283530116081238 h_nor[8]
   - 0.4036439955234528 tau_in[7] - 0.5536800026893616 tau_out[7]
   + 0.1175940036773682 m[7] + z_time_8_layer_0[0] - s_time_8_layer_0[0]
   = -0.1505116222022764
 state_time_step_8_output_layer[0]: tau_in[8]
   - 0.9604629874229431 z_time_8_layer_0[0] = -0.0368359982967377
 state_time_step_8_output_layer[1]: tau_out[8]
   - 0.9853600263595581 z_time_8_layer_0[0] = -0.024433
 state_time_step_8_output_layer[2]: m[8]
   + 0.3904860019683838 z_time_8_layer_0[0] = 0.4039709866046906
 state_time_step_9_layer_0[0]: - 0.00659500015899539 h_nor[0]
   - 0.000658 h_nor[1] - 0.0112509997561574 h_nor[2] + 0.005097 h_nor[3]
   + 0.0122410003095865 h_nor[4] - 0.00444999989122152 h_nor[5]
   - 0.0432949997484684 h_nor[6] - 0.255266010761261 h_nor[7]
   - 0.6405069828033447 h_nor[8] - 0.4283530116081238 h_nor[9]
   - 0.4036439955234528 tau_in[8] - 0.5536800026893616 tau_out[8]
   + 0.1175940036773682 m[8] + z_time_9_layer_0[0] - s_time_9_layer_0[0]
   = -0.1702301116225239
 state_time_step_9_output_layer[0]: tau_in[9]
   - 0.9604629874229431 z_time_9_layer_0[0] = -0.0368359982967377
 state_time_step_9_output_layer[1]: tau_out[9]
   - 0.9853600263595581 z_time_9_layer_0[0] = -0.024433
 state_time_step_9_output_layer[2]: m[9]
   + 0.3904860019683838 z_time_9_layer_0[0] = 0.4039709866046906
 state_time_step_10_layer_0[0]: 0.0186950005590916 h_nor[0]
   - 0.00659500015899539 h_nor[1] - 0.000658 h_nor[2]
   - 0.0112509997561574 h_nor[3] + 0.005097 h_nor[4]
   + 0.0122410003095865 h_nor[5] - 0.00444999989122152 h_nor[6]
   - 0.0432949997484684 h_nor[7] - 0.255266010761261 h_nor[8]
   - 0.6405069828033447 h_nor[9] - 0.4283530116081238 h_nor[10]
   - 0.4036439955234528 tau_in[9] - 0.5536800026893616 tau_out[9]
   + 0.1175940036773682 m[9] + z_time_10_layer_0[0] - s_time_10_layer_0[0]
   = -0.1790038406552483
 state_time_step_10_output_layer[0]: tau_in[10]
   - 0.9604629874229431 z_time_10_layer_0[0] = -0.0368359982967377
 state_time_step_10_output_layer[1]: tau_out[10]
   - 0.9853600263595581 z_time_10_layer_0[0] = -0.024433
 state_time_step_10_output_layer[2]: m[10]
   + 0.3904860019683838 z_time_10_layer_0[0] = 0.4039709866046906
 state_time_step_11_layer_0[0]: 0.0186950005590916 h_nor[1]
   - 0.00659500015899539 h_nor[2] - 0.000658 h_nor[3]
   - 0.0112509997561574 h_nor[4] + 0.005097 h_nor[5]
   + 0.0122410003095865 h_nor[6] - 0.00444999989122152 h_nor[7]
   - 0.0432949997484684 h_nor[8] - 0.255266010761261 h_nor[9]
   - 0.6405069828033447 h_nor[10] - 0.4283530116081238 h_nor[11]
   - 0.4036439955234528 tau_in[10] - 0.5536800026893616 tau_out[10]
   + 0.1175940036773682 m[10] + z_time_11_layer_0[0] - s_time_11_layer_0[0]
   = -0.1859506262795082
 state_time_step_11_output_layer[0]: tau_in[11]
   - 0.9604629874229431 z_time_11_layer_0[0] = -0.0368359982967377
 state_time_step_11_output_layer[1]: tau_out[11]
   - 0.9853600263595581 z_time_11_layer_0[0] = -0.024433
 state_time_step_11_output_layer[2]: m[11]
   + 0.3904860019683838 z_time_11_layer_0[0] = 0.4039709866046906
 output_time_step_0_layer_0[0]: 0.7088149785995483 h_nor[0]
   - 0.2574790120124817 tau_in[0] + 0.2504059970378876 tau_out[0]
   - 0.002571 m[0] + z_time_0_layer_0[0] - s_time_0_layer_0[0]
   = 0.6880507286158744
 output_time_step_0_output_layer[0]: y[0]
   + 1.086115956306458 z_time_0_layer_0[0] = 0.8345879912376404
 output_time_step_1_layer_0[0]: 0.146930992603302 h_nor[0]
   + 0.7088149785995483 h_nor[1] - 0.2574790120124817 tau_in[1]
   + 0.2504059970378876 tau_out[1] - 0.002571 m[1] + z_time_1_layer_0[0]
   - s_time_1_layer_0[0] = 0.7237353106430054
 output_time_step_1_output_layer[0]: y[1]
   + 1.086115956306458 z_time_1_layer_0[0] = 0.8345879912376404
 output_time_step_2_layer_0[0]: 0.0472089983522892 h_nor[0]
   + 0.146930992603302 h_nor[1] + 0.7088149785995483 h_nor[2]
   - 0.2574790120124817 tau_in[2] + 0.2504059970378876 tau_out[2]
   - 0.002571 m[2] + z_time_2_layer_0[0] - s_time_2_layer_0[0]
   = 0.7384917708635945
 output_time_step_2_output_layer[0]: y[2]
   + 1.086115956306458 z_time_2_layer_0[0] = 0.8345879912376404
 output_time_step_3_layer_0[0]: 0.0272279996424913 h_nor[0]
   + 0.0472089983522892 h_nor[1] + 0.146930992603302 h_nor[2]
   + 0.7088149785995483 h_nor[3] - 0.2574790120124817 tau_in[3]
   + 0.2504059970378876 tau_out[3] - 0.002571 m[3] + z_time_3_layer_0[0]
   - s_time_3_layer_0[0] = 0.7447185334994745
 output_time_step_3_output_layer[0]: y[3]
   + 1.086115956306458 z_time_3_layer_0[0] = 0.8345879912376404
 output_time_step_4_layer_0[0]: 0.001302 h_nor[0]
   + 0.0272279996424913 h_nor[1] + 0.0472089983522892 h_nor[2]
   + 0.146930992603302 h_nor[3] + 0.7088149785995483 h_nor[4]
   - 0.2574790120124817 tau_in[4] + 0.2504059970378876 tau_out[4]
   - 0.002571 m[4] + z_time_4_layer_0[0] - s_time_4_layer_0[0]
   = 0.7460891021675534
 output_time_step_4_output_layer[0]: y[4]
   + 1.086115956306458 z_time_4_layer_0[0] = 0.8345879912376404
 output_time_step_5_layer_0[0]: - 0.000463 h_nor[0] + 0.001302 h_nor[1]
   + 0.0272279996424913 h_nor[2] + 0.0472089983522892 h_nor[3]
   + 0.146930992603302 h_nor[4] + 0.7088149785995483 h_nor[5]
   - 0.2574790120124817 tau_in[5] + 0.2504059970378876 tau_out[5]
   - 0.002571 m[5] + z_time_5_layer_0[0] - s_time_5_layer_0[0]
   = 0.7449867757204198
 output_time_step_5_output_layer[0]: y[5]
   + 1.086115956306458 z_time_5_layer_0[0] = 0.8345879912376404
 output_time_step_6_layer_0[0]: - 0.0202920008450747 h_nor[0]
   - 0.000463 h_nor[1] + 0.001302 h_nor[2] + 0.0272279996424913 h_nor[3]
   + 0.0472089983522892 h_nor[4] + 0.146930992603302 h_nor[5]
   + 0.7088149785995483 h_nor[6] - 0.2574790120124817 tau_in[6]
   + 0.2504059970378876 tau_out[6] - 0.002571 m[6] + z_time_6_layer_0[0]
   - s_time_6_layer_0[0] = 0.7359653468685602
 output_time_step_6_output_layer[0]: y[6]
   + 1.086115956306458 z_time_6_layer_0[0] = 0.8345879912376404
 output_time_step_7_layer_0[0]: - 0.0110029997304082 h_nor[0]
   - 0.0202920008450747 h_nor[1] - 0.000463 h_nor[2] + 0.001302 h_nor[3]
   + 0.0272279996424913 h_nor[4] + 0.0472089983522892 h_nor[5]
   + 0.146930992603302 h_nor[6] + 0.7088149785995483 h_nor[7]
   - 0.2574790120124817 tau_in[7] + 0.2504059970378876 tau_out[7]
   - 0.002571 m[7] + z_time_7_layer_0[0] - s_time_7_layer_0[0]
   = 0.7330863069415232
 output_time_step_7_output_layer[0]: y[7]
   + 1.086115956306458 z_time_7_layer_0[0] = 0.8345879912376404
 output_time_step_8_layer_0[0]: 0.008541 h_nor[0]
   - 0.0110029997304082 h_nor[1] - 0.0202920008450747 h_nor[2]
   - 0.000463 h_nor[3] + 0.001302 h_nor[4] + 0.0272279996424913 h_nor[5]
   + 0.0472089983522892 h_nor[6] + 0.146930992603302 h_nor[7]
   + 0.7088149785995483 h_nor[8] - 0.2574790120124817 tau_in[8]
   + 0.2504059970378876 tau_out[8] - 0.002571 m[8] + z_time_8_layer_0[0]
   - s_time_8_layer_0[0] = 0.7315083464001697
 output_time_step_8_output_layer[0]: y[8]
   + 1.086115956306458 z_time_8_layer_0[0] = 0.8345879912376404
 output_time_step_9_layer_0[0]: 0.003302 h_nor[0] + 0.008541 h_nor[1]
   - 0.0110029997304082 h_nor[2] - 0.0202920008450747 h_nor[3]
   - 0.000463 h_nor[4] + 0.001302 h_nor[5] + 0.0272279996424913 h_nor[6]
   + 0.0472089983522892 h_nor[7] + 0.146930992603302 h_nor[8]
   + 0.7088149785995483 h_nor[9] - 0.2574790120124817 tau_in[9]
   + 0.2504059970378876 tau_out[9] - 0.002571 m[9] + z_time_9_layer_0[0]
   - s_time_9_layer_0[0] = 0.7330884987359941
 output_time_step_9_output_layer[0]: y[9]
   + 1.086115956306458 z_time_9_layer_0[0] = 0.8345879912376404
 output_time_step_10_layer_0[0]: 0.0387040004134178 h_nor[0]
   + 0.003302 h_nor[1] + 0.008541 h_nor[2] - 0.0110029997304082 h_nor[3]
   - 0.0202920008450747 h_nor[4] - 0.000463 h_nor[5] + 0.001302 h_nor[6]
   + 0.0272279996424913 h_nor[7] + 0.0472089983522892 h_nor[8]
   + 0.146930992603302 h_nor[9] + 0.7088149785995483 h_nor[10]
   - 0.2574790120124817 tau_in[10] + 0.2504059970378876 tau_out[10]
   - 0.002571 m[10] + z_time_10_layer_0[0] - s_time_10_layer_0[0]
   = 0.7423290014266968
 output_time_step_10_output_layer[0]: y[10]
   + 1.086115956306458 z_time_10_layer_0[0] = 0.8345879912376404
 output_time_step_11_layer_0[0]: 0.0387040004134178 h_nor[1]
   + 0.003302 h_nor[2] + 0.008541 h_nor[3] - 0.0110029997304082 h_nor[4]
   - 0.0202920008450747 h_nor[5] - 0.000463 h_nor[6] + 0.001302 h_nor[7]
   + 0.0272279996424913 h_nor[8] + 0.0472089983522892 h_nor[9]
   + 0.146930992603302 h_nor[10] + 0.7088149785995483 h_nor[11]
   - 0.2574790120124817 tau_in[11] + 0.2504059970378876 tau_out[11]
   - 0.002571 m[11] + z_time_11_layer_0[0] - s_time_11_layer_0[0]
   = 0.7423290014266968
 output_time_step_11_output_layer[0]: y[11]
   + 1.086115956306458 z_time_11_layer_0[0] = 0.8345879912376404
 state_time_step_0indicator_constraint_z_0: a_time_0_layer_0[0] = 1 ->
   z_time_0_layer_0[0] <= 0
 state_time_step_0indicator_constraint_s_0: a_time_0_layer_0[0] = 0 ->
   s_time_0_layer_0[0] <= 0
 state_time_step_1indicator_constraint_z_0: a_time_1_layer_0[0] = 1 ->
   z_time_1_layer_0[0] <= 0
 state_time_step_1indicator_constraint_s_0: a_time_1_layer_0[0] = 0 ->
   s_time_1_layer_0[0] <= 0
 state_time_step_2indicator_constraint_z_0: a_time_2_layer_0[0] = 1 ->
   z_time_2_layer_0[0] <= 0
 state_time_step_2indicator_constraint_s_0: a_time_2_layer_0[0] = 0 ->
   s_time_2_layer_0[0] <= 0
 state_time_step_3indicator_constraint_z_0: a_time_3_layer_0[0] = 1 ->
   z_time_3_layer_0[0] <= 0
 state_time_step_3indicator_constraint_s_0: a_time_3_layer_0[0] = 0 ->
   s_time_3_layer_0[0] <= 0
 state_time_step_4indicator_constraint_z_0: a_time_4_layer_0[0] = 1 ->
   z_time_4_layer_0[0] <= 0
 state_time_step_4indicator_constraint_s_0: a_time_4_layer_0[0] = 0 ->
   s_time_4_layer_0[0] <= 0
 state_time_step_5indicator_constraint_z_0: a_time_5_layer_0[0] = 1 ->
   z_time_5_layer_0[0] <= 0
 state_time_step_5indicator_constraint_s_0: a_time_5_layer_0[0] = 0 ->
   s_time_5_layer_0[0] <= 0
 state_time_step_6indicator_constraint_z_0: a_time_6_layer_0[0] = 1 ->
   z_time_6_layer_0[0] <= 0
 state_time_step_6indicator_constraint_s_0: a_time_6_layer_0[0] = 0 ->
   s_time_6_layer_0[0] <= 0
 state_time_step_7indicator_constraint_z_0: a_time_7_layer_0[0] = 1 ->
   z_time_7_layer_0[0] <= 0
 state_time_step_7indicator_constraint_s_0: a_time_7_layer_0[0] = 0 ->
   s_time_7_layer_0[0] <= 0
 state_time_step_8indicator_constraint_z_0: a_time_8_layer_0[0] = 1 ->
   z_time_8_layer_0[0] <= 0
 state_time_step_8indicator_constraint_s_0: a_time_8_layer_0[0] = 0 ->
   s_time_8_layer_0[0] <= 0
 state_time_step_9indicator_constraint_z_0: a_time_9_layer_0[0] = 1 ->
   z_time_9_layer_0[0] <= 0
 state_time_step_9indicator_constraint_s_0: a_time_9_layer_0[0] = 0 ->
   s_time_9_layer_0[0] <= 0
 state_time_step_10indicator_constraint_z_0: a_time_10_layer_0[0] = 1 ->
   z_time_10_layer_0[0] <= 0
 state_time_step_10indicator_constraint_s_0: a_time_10_layer_0[0] = 0 ->
   s_time_10_layer_0[0] <= 0
 state_time_step_11indicator_constraint_z_0: a_time_11_layer_0[0] = 1 ->
   z_time_11_layer_0[0] <= 0
 state_time_step_11indicator_constraint_s_0: a_time_11_layer_0[0] = 0 ->
   s_time_11_layer_0[0] <= 0
 output_time_step_0indicator_constraint_z_0: a_time_0_layer_0[0] = 1 ->
   z_time_0_layer_0[0] <= 0
 output_time_step_0indicator_constraint_s_0: a_time_0_layer_0[0] = 0 ->
   s_time_0_layer_0[0] <= 0
 output_time_step_1indicator_constraint_z_0: a_time_1_layer_0[0] = 1 ->
   z_time_1_layer_0[0] <= 0
 output_time_step_1indicator_constraint_s_0: a_time_1_layer_0[0] = 0 ->
   s_time_1_layer_0[0] <= 0
 output_time_step_2indicator_constraint_z_0: a_time_2_layer_0[0] = 1 ->
   z_time_2_layer_0[0] <= 0
 output_time_step_2indicator_constraint_s_0: a_time_2_layer_0[0] = 0 ->
   s_time_2_layer_0[0] <= 0
 output_time_step_3indicator_constraint_z_0: a_time_3_layer_0[0] = 1 ->
   z_time_3_layer_0[0] <= 0
 output_time_step_3indicator_constraint_s_0: a_time_3_layer_0[0] = 0 ->
   s_time_3_layer_0[0] <= 0
 output_time_step_4indicator_constraint_z_0: a_time_4_layer_0[0] = 1 ->
   z_time_4_layer_0[0] <= 0
 output_time_step_4indicator_constraint_s_0: a_time_4_layer_0[0] = 0 ->
   s_time_4_layer_0[0] <= 0
 output_time_step_5indicator_constraint_z_0: a_time_5_layer_0[0] = 1 ->
   z_time_5_layer_0[0] <= 0
 output_time_step_5indicator_constraint_s_0: a_time_5_layer_0[0] = 0 ->
   s_time_5_layer_0[0] <= 0
 output_time_step_6indicator_constraint_z_0: a_time_6_layer_0[0] = 1 ->
   z_time_6_layer_0[0] <= 0
 output_time_step_6indicator_constraint_s_0: a_time_6_layer_0[0] = 0 ->
   s_time_6_layer_0[0] <= 0
 output_time_step_7indicator_constraint_z_0: a_time_7_layer_0[0] = 1 ->
   z_time_7_layer_0[0] <= 0
 output_time_step_7indicator_constraint_s_0: a_time_7_layer_0[0] = 0 ->
   s_time_7_layer_0[0] <= 0
 output_time_step_8indicator_constraint_z_0: a_time_8_layer_0[0] = 1 ->
   z_time_8_layer_0[0] <= 0
 output_time_step_8indicator_constraint_s_0: a_time_8_layer_0[0] = 0 ->
   s_time_8_layer_0[0] <= 0
 output_time_step_9indicator_constraint_z_0: a_time_9_layer_0[0] = 1 ->
   z_time_9_layer_0[0] <= 0
 output_time_step_9indicator_constraint_s_0: a_time_9_layer_0[0] = 0 ->
   s_time_9_layer_0[0] <= 0
 output_time_step_10indicator_constraint_z_0: a_time_10_layer_0[0] = 1 ->
   z_time_10_layer_0[0] <= 0
 output_time_step_10indicator_constraint_s_0: a_time_10_layer_0[0] = 0 ->
   s_time_10_layer_0[0] <= 0
 output_time_step_11indicator_constraint_z_0: a_time_11_layer_0[0] = 1 ->
   z_time_11_layer_0[0] <= 0
 output_time_step_11indicator_constraint_s_0: a_time_11_layer_0[0] = 0 ->
   s_time_11_layer_0[0] <= 0
Bounds
 h[0] <= 70
 h[1] <= 70
 h[2] <= 70
 h[3] <= 70
 h[4] <= 70
 h[5] <= 70
 h[6] <= 70
 h[7] <= 70
 h[8] <= 70
 h[9] <= 70
 h[10] <= 70
 h[11] <= 70
 5 <= p[0] <= 50
 5 <= p[1] <= 50
 5 <= p[2] <= 50
 5 <= p[3] <= 50
 5 <= p[4] <= 50
 5 <= p[5] <= 50
 5 <= p[6] <= 50
 5 <= p[7] <= 50
 5 <= p[8] <= 50
 5 <= p[9] <= 50
 5 <= p[10] <= 50
 5 <= p[11] <= 50
 h_nor[0] <= 1
 h_nor[1] <= 1
 h_nor[2] <= 1
 h_nor[3] <= 1
 h_nor[4] <= 1
 h_nor[5] <= 1
 h_nor[6] <= 1
 h_nor[7] <= 1
 h_nor[8] <= 1
 h_nor[9] <= 1
 h_nor[10] <= 1
 h_nor[11] <= 1
 tau_in[0] <= 1
 tau_in[1] <= 1
 tau_in[2] <= 1
 tau_in[3] <= 1
 tau_in[4] <= 1
 tau_in[5] <= 1
 tau_in[6] <= 1
 tau_in[7] <= 1
 tau_in[8] <= 1
 tau_in[9] <= 1
 tau_in[10] <= 1
 tau_in[11] <= 1
 -0.104208 <= tau_out[0] <= 1
 -0.104208 <= tau_out[1] <= 1
 -0.104208 <= tau_out[2] <= 1
 -0.104208 <= tau_out[3] <= 1
 -0.104208 <= tau_out[4] <= 1
 -0.104208 <= tau_out[5] <= 1
 -0.104208 <= tau_out[6] <= 1
 -0.104208 <= tau_out[7] <= 1
 -0.104208 <= tau_out[8] <= 1
 -0.104208 <= tau_out[9] <= 1
 -0.104208 <= tau_out[10] <= 1
 -0.104208 <= tau_out[11] <= 1
 m[0] <= 1.172853
 m[1] <= 1.172853
 m[2] <= 1.172853
 m[3] <= 1.172853
 m[4] <= 1.172853
 m[5] <= 1.172853
 m[6] <= 1.172853
 m[7] <= 1.172853
 m[8] <= 1.172853
 m[9] <= 1.172853
 m[10] <= 1.172853
 m[11] <= 1.172853
Binaries
 a_time_0_layer_0[0] a_time_1_layer_0[0] a_time_2_layer_0[0]
 a_time_3_layer_0[0] a_time_4_layer_0[0] a_time_5_layer_0[0]
 a_time_6_layer_0[0] a_time_7_layer_0[0] a_time_8_layer_0[0]
 a_time_9_layer_0[0] a_time_10_layer_0[0] a_time_11_layer_0[0]
 a_time_0_layer_0[0] a_time_1_layer_0[0] a_time_2_layer_0[0]
 a_time_3_layer_0[0] a_time_4_layer_0[0] a_time_5_layer_0[0]
 a_time_6_layer_0[0] a_time_7_layer_0[0] a_time_8_layer_0[0]
 a_time_9_layer_0[0] a_time_10_layer_0[0] a_time_11_layer_0[0]
End
