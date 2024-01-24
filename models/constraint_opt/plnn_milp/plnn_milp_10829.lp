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
 delivered_heat_L[0]: y[0] <= 0.5033691190401008
 delivered_heat_L[1]: y[1] <= 0.4989485317335017
 delivered_heat_L[2]: y[2] <= 0.5041585096305652
 delivered_heat_L[3]: y[3] <= 0.5139469529523208
 delivered_heat_L[4]: y[4] <= 0.5360498894853173
 delivered_heat_L[5]: y[5] <= 0.5453646984527944
 delivered_heat_L[6]: y[6] <= 0.5453646984527944
 delivered_heat_L[7]: y[7] <= 0.5362077676034102
 delivered_heat_L[8]: y[8] <= 0.5169466371960847
 delivered_heat_L[9]: y[9] <= 0.4656362488159141
 delivered_heat_L[10]: y[10] <= 0.3628575939374803
 delivered_heat_L[11]: y[11] <= 0.2919703189137985
 delivered_heat_G[0]: y[0] >= 0.4833691190401009
 delivered_heat_G[1]: y[1] >= 0.4789485317335017
 delivered_heat_G[2]: y[2] >= 0.4841585096305652
 delivered_heat_G[3]: y[3] >= 0.4939469529523208
 delivered_heat_G[4]: y[4] >= 0.5160498894853173
 delivered_heat_G[5]: y[5] >= 0.5253646984527944
 delivered_heat_G[6]: y[6] >= 0.5253646984527944
 delivered_heat_G[7]: y[7] >= 0.5162077676034101
 delivered_heat_G[8]: y[8] >= 0.4969466371960847
 delivered_heat_G[9]: y[9] >= 0.4456362488159141
 delivered_heat_G[10]: y[10] >= 0.3428575939374803
 delivered_heat_G[11]: y[11] >= 0.2719703189137985
 state_time_step_0_layer_0[0]: - 0.4283530116081238 h_nor[0]
   + z_time_0_layer_0[0] - s_time_0_layer_0[0] = 0.1141133891603716
 state_time_step_0_output_layer[0]: tau_in[0]
   - 0.9604629874229431 z_time_0_layer_0[0] = -0.0368359982967377
 state_time_step_0_output_layer[1]: tau_out[0]
   - 0.9853600263595581 z_time_0_layer_0[0] = -0.024433
 state_time_step_0_output_layer[2]: m[0]
   + 0.3904860019683838 z_time_0_layer_0[0] = 0.4039709866046906
 state_time_step_1_layer_0[0]: - 0.6405069828033447 h_nor[0]
   - 0.4283530116081238 h_nor[1] - 0.4036439955234528 tau_in[0]
   - 0.5536800026893616 tau_out[0] + 0.1175940036773682 m[0]
   + z_time_1_layer_0[0] - s_time_1_layer_0[0] = -0.3957419265266106
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
   = -0.5126775260691049
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
   + z_time_3_layer_0[0] - s_time_3_layer_0[0] = -0.5440857690816531
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
   = -0.5658349598394746
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
   + z_time_5_layer_0[0] - s_time_5_layer_0[0] = -0.5837214316094441
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
   = -0.5899763308059998
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
   + z_time_7_layer_0[0] - s_time_7_layer_0[0] = -0.590916928873123
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
   = -0.5746949141339528
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
   = -0.5384313398082626
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
   = -0.4404243234466181
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
   = -0.3175343339742265
 state_time_step_11_output_layer[0]: tau_in[11]
   - 0.9604629874229431 z_time_11_layer_0[0] = -0.0368359982967377
 state_time_step_11_output_layer[1]: tau_out[11]
   - 0.9853600263595581 z_time_11_layer_0[0] = -0.024433
 state_time_step_11_output_layer[2]: m[11]
   + 0.3904860019683838 z_time_11_layer_0[0] = 0.4039709866046906
 output_time_step_0_layer_0[0]: 0.7088149785995483 h_nor[0]
   - 0.2574790120124817 tau_in[0] + 0.2504059970378876 tau_out[0]
   - 0.002571 m[0] + z_time_0_layer_0[0] - s_time_0_layer_0[0]
   = 0.6419246448418235
 output_time_step_0_output_layer[0]: y[0]
   + 1.086115956306458 z_time_0_layer_0[0] = 0.8345879912376404
 output_time_step_1_layer_0[0]: 0.146930992603302 h_nor[0]
   + 0.7088149785995483 h_nor[1] - 0.2574790120124817 tau_in[1]
   + 0.2504059970378876 tau_out[1] - 0.002571 m[1] + z_time_1_layer_0[0]
   - s_time_1_layer_0[0] = 0.7084105959296928
 output_time_step_1_output_layer[0]: y[1]
   + 1.086115956306458 z_time_1_layer_0[0] = 0.8345879912376404
 output_time_step_2_layer_0[0]: 0.0472089983522892 h_nor[0]
   + 0.146930992603302 h_nor[1] + 0.7088149785995483 h_nor[2]
   - 0.2574790120124817 tau_in[2] + 0.2504059970378876 tau_out[2]
   - 0.002571 m[2] + z_time_2_layer_0[0] - s_time_2_layer_0[0]
   = 0.7263184047770921
 output_time_step_2_output_layer[0]: y[2]
   + 1.086115956306458 z_time_2_layer_0[0] = 0.8345879912376404
 output_time_step_3_layer_0[0]: 0.0272279996424913 h_nor[0]
   + 0.0472089983522892 h_nor[1] + 0.146930992603302 h_nor[2]
   + 0.7088149785995483 h_nor[3] - 0.2574790120124817 tau_in[3]
   + 0.2504059970378876 tau_out[3] - 0.002571 m[3] + z_time_3_layer_0[0]
   - s_time_3_layer_0[0] = 0.7315220440511103
 output_time_step_3_output_layer[0]: y[3]
   + 1.086115956306458 z_time_3_layer_0[0] = 0.8345879912376404
 output_time_step_4_layer_0[0]: 0.001302 h_nor[0]
   + 0.0272279996424913 h_nor[1] + 0.0472089983522892 h_nor[2]
   + 0.146930992603302 h_nor[3] + 0.7088149785995483 h_nor[4]
   - 0.2574790120124817 tau_in[4] + 0.2504059970378876 tau_out[4]
   - 0.002571 m[4] + z_time_4_layer_0[0] - s_time_4_layer_0[0]
   = 0.7287566758383548
 output_time_step_4_output_layer[0]: y[4]
   + 1.086115956306458 z_time_4_layer_0[0] = 0.8345879912376404
 output_time_step_5_layer_0[0]: - 0.000463 h_nor[0] + 0.001302 h_nor[1]
   + 0.0272279996424913 h_nor[2] + 0.0472089983522892 h_nor[3]
   + 0.146930992603302 h_nor[4] + 0.7088149785995483 h_nor[5]
   - 0.2574790120124817 tau_in[5] + 0.2504059970378876 tau_out[5]
   - 0.002571 m[5] + z_time_5_layer_0[0] - s_time_5_layer_0[0]
   = 0.7295405017020506
 output_time_step_5_output_layer[0]: y[5]
   + 1.086115956306458 z_time_5_layer_0[0] = 0.8345879912376404
 output_time_step_6_layer_0[0]: - 0.0202920008450747 h_nor[0]
   - 0.000463 h_nor[1] + 0.001302 h_nor[2] + 0.0272279996424913 h_nor[3]
   + 0.0472089983522892 h_nor[4] + 0.146930992603302 h_nor[5]
   + 0.7088149785995483 h_nor[6] - 0.2574790120124817 tau_in[6]
   + 0.2504059970378876 tau_out[6] - 0.002571 m[6] + z_time_6_layer_0[0]
   - s_time_6_layer_0[0] = 0.7209010573940375
 output_time_step_6_output_layer[0]: y[6]
   + 1.086115956306458 z_time_6_layer_0[0] = 0.8345879912376404
 output_time_step_7_layer_0[0]: - 0.0110029997304082 h_nor[0]
   - 0.0202920008450747 h_nor[1] - 0.000463 h_nor[2] + 0.001302 h_nor[3]
   + 0.0272279996424913 h_nor[4] + 0.0472089983522892 h_nor[5]
   + 0.146930992603302 h_nor[6] + 0.7088149785995483 h_nor[7]
   - 0.2574790120124817 tau_in[7] + 0.2504059970378876 tau_out[7]
   - 0.002571 m[7] + z_time_7_layer_0[0] - s_time_7_layer_0[0]
   = 0.7172881781378085
 output_time_step_7_output_layer[0]: y[7]
   + 1.086115956306458 z_time_7_layer_0[0] = 0.8345879912376404
 output_time_step_8_layer_0[0]: 0.008541 h_nor[0]
   - 0.0110029997304082 h_nor[1] - 0.0202920008450747 h_nor[2]
   - 0.000463 h_nor[3] + 0.001302 h_nor[4] + 0.0272279996424913 h_nor[5]
   + 0.0472089983522892 h_nor[6] + 0.146930992603302 h_nor[7]
   + 0.7088149785995483 h_nor[8] - 0.2574790120124817 tau_in[8]
   + 0.2504059970378876 tau_out[8] - 0.002571 m[8] + z_time_8_layer_0[0]
   - s_time_8_layer_0[0] = 0.7220295330853486
 output_time_step_8_output_layer[0]: y[8]
   + 1.086115956306458 z_time_8_layer_0[0] = 0.8345879912376404
 output_time_step_9_layer_0[0]: 0.003302 h_nor[0] + 0.008541 h_nor[1]
   - 0.0110029997304082 h_nor[2] - 0.0202920008450747 h_nor[3]
   - 0.000463 h_nor[4] + 0.001302 h_nor[5] + 0.0272279996424913 h_nor[6]
   + 0.0472089983522892 h_nor[7] + 0.146930992603302 h_nor[8]
   + 0.7088149785995483 h_nor[9] - 0.2574790120124817 tau_in[9]
   + 0.2504059970378876 tau_out[9] - 0.002571 m[9] + z_time_9_layer_0[0]
   - s_time_9_layer_0[0] = 0.7244878118761257
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
