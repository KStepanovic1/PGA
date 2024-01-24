\ Model plnn_milp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  8.1817 h[0] + 8.1817 h[1] + 8.1817 h[2] + 8.1817 h[3] + 8.1817 h[4]
   + 8.1817 h[5] + 8.1817 h[6] + 8.1817 h[7] + 8.1817 h[8] + 8.1817 h[9]
   + 8.1817 h[10] + 8.1817 h[11] + 0 p[0] + 0 p[1] + 0 p[2] + 0 p[3]
   + 0 p[4] + 0 p[5] + 0 p[6] + 0 p[7] + 0 p[8] + 0 p[9] + 0 p[10]
   + 0 p[11] + 0 a_time_0_layer_0[0] + 0 a_time_0_layer_1[0]
   + 0 a_time_1_layer_0[0] + 0 a_time_1_layer_1[0] + 0 a_time_2_layer_0[0]
   + 0 a_time_2_layer_1[0] + 0 a_time_3_layer_0[0] + 0 a_time_3_layer_1[0]
   + 0 a_time_4_layer_0[0] + 0 a_time_4_layer_1[0] + 0 a_time_5_layer_0[0]
   + 0 a_time_5_layer_1[0] + 0 a_time_6_layer_0[0] + 0 a_time_6_layer_1[0]
   + 0 a_time_7_layer_0[0] + 0 a_time_7_layer_1[0] + 0 a_time_8_layer_0[0]
   + 0 a_time_8_layer_1[0] + 0 a_time_9_layer_0[0] + 0 a_time_9_layer_1[0]
   + 0 a_time_10_layer_0[0] + 0 a_time_10_layer_1[0]
   + 0 a_time_11_layer_0[0] + 0 a_time_11_layer_1[0]
   + 0 a_time_0_layer_0[0] + 0 a_time_0_layer_1[0] + 0 a_time_1_layer_0[0]
   + 0 a_time_1_layer_1[0] + 0 a_time_2_layer_0[0] + 0 a_time_2_layer_1[0]
   + 0 a_time_3_layer_0[0] + 0 a_time_3_layer_1[0] + 0 a_time_4_layer_0[0]
   + 0 a_time_4_layer_1[0] + 0 a_time_5_layer_0[0] + 0 a_time_5_layer_1[0]
   + 0 a_time_6_layer_0[0] + 0 a_time_6_layer_1[0] + 0 a_time_7_layer_0[0]
   + 0 a_time_7_layer_1[0] + 0 a_time_8_layer_0[0] + 0 a_time_8_layer_1[0]
   + 0 a_time_9_layer_0[0] + 0 a_time_9_layer_1[0] + 0 a_time_10_layer_0[0]
   + 0 a_time_10_layer_1[0] + 0 a_time_11_layer_0[0]
   + 0 a_time_11_layer_1[0]
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
 delivered_heat_L[0]: y[0] <= 0.1928228607514998
 delivered_heat_L[1]: y[1] <= 0.2051373539627408
 delivered_heat_L[2]: y[2] <= 0.2221881907167667
 delivered_heat_L[3]: y[3] <= 0.2423965898326491
 delivered_heat_L[4]: y[4] <= 0.2521850331544048
 delivered_heat_L[5]: y[5] <= 0.250132617619198
 delivered_heat_L[6]: y[6] <= 0.2447647616040417
 delivered_heat_L[7]: y[7] <= 0.2141364066940322
 delivered_heat_L[8]: y[8] <= 0.157773918534891
 delivered_heat_L[9]: y[9] <= 0.1108841174613198
 delivered_heat_L[10]: y[10] <= 0.0722039785285759
 delivered_heat_L[11]: y[11] <= 0.0845184717398168
 delivered_heat_G[0]: y[0] >= 0.1728228607514998
 delivered_heat_G[1]: y[1] >= 0.1851373539627408
 delivered_heat_G[2]: y[2] >= 0.2021881907167667
 delivered_heat_G[3]: y[3] >= 0.2223965898326491
 delivered_heat_G[4]: y[4] >= 0.2321850331544048
 delivered_heat_G[5]: y[5] >= 0.230132617619198
 delivered_heat_G[6]: y[6] >= 0.2247647616040417
 delivered_heat_G[7]: y[7] >= 0.1941364066940322
 delivered_heat_G[8]: y[8] >= 0.137773918534891
 delivered_heat_G[9]: y[9] >= 0.0908841174613198
 delivered_heat_G[10]: y[10] >= 0.0522039785285759
 delivered_heat_G[11]: y[11] >= 0.0645184717398168
 state_time_step_0_layer_0[0]: z_time_0_layer_0[0] - s_time_0_layer_0[0]
   = 0
 state_time_step_0_layer_1[0]: z_time_0_layer_1[0] - s_time_0_layer_1[0]
   = 0
 state_time_step_0_output_layer[0]: tau_in[0] = 0.4249140024185181
 state_time_step_0_output_layer[1]: tau_out[0] = 0.4373010098934174
 state_time_step_0_output_layer[2]: m[0] = 0.2183790057897568
 state_time_step_1_layer_0[0]: z_time_1_layer_0[0] - s_time_1_layer_0[0]
   = 0
 state_time_step_1_layer_1[0]: z_time_1_layer_1[0] - s_time_1_layer_1[0]
   = 0
 state_time_step_1_output_layer[0]: tau_in[1] = 0.4249140024185181
 state_time_step_1_output_layer[1]: tau_out[1] = 0.4373010098934174
 state_time_step_1_output_layer[2]: m[1] = 0.2183790057897568
 state_time_step_2_layer_0[0]: z_time_2_layer_0[0] - s_time_2_layer_0[0]
   = 0
 state_time_step_2_layer_1[0]: z_time_2_layer_1[0] - s_time_2_layer_1[0]
   = 0
 state_time_step_2_output_layer[0]: tau_in[2] = 0.4249140024185181
 state_time_step_2_output_layer[1]: tau_out[2] = 0.4373010098934174
 state_time_step_2_output_layer[2]: m[2] = 0.2183790057897568
 state_time_step_3_layer_0[0]: z_time_3_layer_0[0] - s_time_3_layer_0[0]
   = 0
 state_time_step_3_layer_1[0]: z_time_3_layer_1[0] - s_time_3_layer_1[0]
   = 0
 state_time_step_3_output_layer[0]: tau_in[3] = 0.4249140024185181
 state_time_step_3_output_layer[1]: tau_out[3] = 0.4373010098934174
 state_time_step_3_output_layer[2]: m[3] = 0.2183790057897568
 state_time_step_4_layer_0[0]: z_time_4_layer_0[0] - s_time_4_layer_0[0]
   = 0
 state_time_step_4_layer_1[0]: z_time_4_layer_1[0] - s_time_4_layer_1[0]
   = 0
 state_time_step_4_output_layer[0]: tau_in[4] = 0.4249140024185181
 state_time_step_4_output_layer[1]: tau_out[4] = 0.4373010098934174
 state_time_step_4_output_layer[2]: m[4] = 0.2183790057897568
 state_time_step_5_layer_0[0]: z_time_5_layer_0[0] - s_time_5_layer_0[0]
   = 0
 state_time_step_5_layer_1[0]: z_time_5_layer_1[0] - s_time_5_layer_1[0]
   = 0
 state_time_step_5_output_layer[0]: tau_in[5] = 0.4249140024185181
 state_time_step_5_output_layer[1]: tau_out[5] = 0.4373010098934174
 state_time_step_5_output_layer[2]: m[5] = 0.2183790057897568
 state_time_step_6_layer_0[0]: z_time_6_layer_0[0] - s_time_6_layer_0[0]
   = 0
 state_time_step_6_layer_1[0]: z_time_6_layer_1[0] - s_time_6_layer_1[0]
   = 0
 state_time_step_6_output_layer[0]: tau_in[6] = 0.4249140024185181
 state_time_step_6_output_layer[1]: tau_out[6] = 0.4373010098934174
 state_time_step_6_output_layer[2]: m[6] = 0.2183790057897568
 state_time_step_7_layer_0[0]: z_time_7_layer_0[0] - s_time_7_layer_0[0]
   = 0
 state_time_step_7_layer_1[0]: z_time_7_layer_1[0] - s_time_7_layer_1[0]
   = 0
 state_time_step_7_output_layer[0]: tau_in[7] = 0.4249140024185181
 state_time_step_7_output_layer[1]: tau_out[7] = 0.4373010098934174
 state_time_step_7_output_layer[2]: m[7] = 0.2183790057897568
 state_time_step_8_layer_0[0]: z_time_8_layer_0[0] - s_time_8_layer_0[0]
   = 0
 state_time_step_8_layer_1[0]: z_time_8_layer_1[0] - s_time_8_layer_1[0]
   = 0
 state_time_step_8_output_layer[0]: tau_in[8] = 0.4249140024185181
 state_time_step_8_output_layer[1]: tau_out[8] = 0.4373010098934174
 state_time_step_8_output_layer[2]: m[8] = 0.2183790057897568
 state_time_step_9_layer_0[0]: z_time_9_layer_0[0] - s_time_9_layer_0[0]
   = 0
 state_time_step_9_layer_1[0]: z_time_9_layer_1[0] - s_time_9_layer_1[0]
   = 0
 state_time_step_9_output_layer[0]: tau_in[9] = 0.4249140024185181
 state_time_step_9_output_layer[1]: tau_out[9] = 0.4373010098934174
 state_time_step_9_output_layer[2]: m[9] = 0.2183790057897568
 state_time_step_10_layer_0[0]: z_time_10_layer_0[0] - s_time_10_layer_0[0]
   = 0
 state_time_step_10_layer_1[0]: z_time_10_layer_1[0] - s_time_10_layer_1[0]
   = 0
 state_time_step_10_output_layer[0]: tau_in[10] = 0.4249140024185181
 state_time_step_10_output_layer[1]: tau_out[10] = 0.4373010098934174
 state_time_step_10_output_layer[2]: m[10] = 0.2183790057897568
 state_time_step_11_layer_0[0]: z_time_11_layer_0[0] - s_time_11_layer_0[0]
   = 0
 state_time_step_11_layer_1[0]: z_time_11_layer_1[0] - s_time_11_layer_1[0]
   = 0
 state_time_step_11_output_layer[0]: tau_in[11] = 0.4249140024185181
 state_time_step_11_output_layer[1]: tau_out[11] = 0.4373010098934174
 state_time_step_11_output_layer[2]: m[11] = 0.2183790057897568
 output_time_step_0_layer_0[0]: z_time_0_layer_0[0] - s_time_0_layer_0[0]
   = 0
 output_time_step_0_layer_1[0]: z_time_0_layer_1[0] - s_time_0_layer_1[0]
   = 0
 output_time_step_0_output_layer[0]: y[0] = 0.3925620019435883
 output_time_step_1_layer_0[0]: z_time_1_layer_0[0] - s_time_1_layer_0[0]
   = 0
 output_time_step_1_layer_1[0]: z_time_1_layer_1[0] - s_time_1_layer_1[0]
   = 0
 output_time_step_1_output_layer[0]: y[1] = 0.3925620019435883
 output_time_step_2_layer_0[0]: z_time_2_layer_0[0] - s_time_2_layer_0[0]
   = 0
 output_time_step_2_layer_1[0]: z_time_2_layer_1[0] - s_time_2_layer_1[0]
   = 0
 output_time_step_2_output_layer[0]: y[2] = 0.3925620019435883
 output_time_step_3_layer_0[0]: z_time_3_layer_0[0] - s_time_3_layer_0[0]
   = 0
 output_time_step_3_layer_1[0]: z_time_3_layer_1[0] - s_time_3_layer_1[0]
   = 0
 output_time_step_3_output_layer[0]: y[3] = 0.3925620019435883
 output_time_step_4_layer_0[0]: z_time_4_layer_0[0] - s_time_4_layer_0[0]
   = 0
 output_time_step_4_layer_1[0]: z_time_4_layer_1[0] - s_time_4_layer_1[0]
   = 0
 output_time_step_4_output_layer[0]: y[4] = 0.3925620019435883
 output_time_step_5_layer_0[0]: z_time_5_layer_0[0] - s_time_5_layer_0[0]
   = 0
 output_time_step_5_layer_1[0]: z_time_5_layer_1[0] - s_time_5_layer_1[0]
   = 0
 output_time_step_5_output_layer[0]: y[5] = 0.3925620019435883
 output_time_step_6_layer_0[0]: z_time_6_layer_0[0] - s_time_6_layer_0[0]
   = 0
 output_time_step_6_layer_1[0]: z_time_6_layer_1[0] - s_time_6_layer_1[0]
   = 0
 output_time_step_6_output_layer[0]: y[6] = 0.3925620019435883
 output_time_step_7_layer_0[0]: z_time_7_layer_0[0] - s_time_7_layer_0[0]
   = 0
 output_time_step_7_layer_1[0]: z_time_7_layer_1[0] - s_time_7_layer_1[0]
   = 0
 output_time_step_7_output_layer[0]: y[7] = 0.3925620019435883
 output_time_step_8_layer_0[0]: z_time_8_layer_0[0] - s_time_8_layer_0[0]
   = 0
 output_time_step_8_layer_1[0]: z_time_8_layer_1[0] - s_time_8_layer_1[0]
   = 0
 output_time_step_8_output_layer[0]: y[8] = 0.3925620019435883
 output_time_step_9_layer_0[0]: z_time_9_layer_0[0] - s_time_9_layer_0[0]
   = 0
 output_time_step_9_layer_1[0]: z_time_9_layer_1[0] - s_time_9_layer_1[0]
   = 0
 output_time_step_9_output_layer[0]: y[9] = 0.3925620019435883
 output_time_step_10_layer_0[0]: z_time_10_layer_0[0]
   - s_time_10_layer_0[0] = 0
 output_time_step_10_layer_1[0]: z_time_10_layer_1[0]
   - s_time_10_layer_1[0] = 0
 output_time_step_10_output_layer[0]: y[10] = 0.3925620019435883
 output_time_step_11_layer_0[0]: z_time_11_layer_0[0]
   - s_time_11_layer_0[0] = 0
 output_time_step_11_layer_1[0]: z_time_11_layer_1[0]
   - s_time_11_layer_1[0] = 0
 output_time_step_11_output_layer[0]: y[11] = 0.3925620019435883
 state_time_step_0indicator_constraint_z_0: a_time_0_layer_0[0] = 1 ->
   z_time_0_layer_0[0] <= 0
 state_time_step_0indicator_constraint_s_0: a_time_0_layer_0[0] = 0 ->
   s_time_0_layer_0[0] <= 0
 state_time_step_0indicator_constraint_z_1: a_time_0_layer_1[0] = 1 ->
   z_time_0_layer_1[0] <= 0
 state_time_step_0indicator_constraint_s_1: a_time_0_layer_1[0] = 0 ->
   s_time_0_layer_1[0] <= 0
 state_time_step_1indicator_constraint_z_0: a_time_1_layer_0[0] = 1 ->
   z_time_1_layer_0[0] <= 0
 state_time_step_1indicator_constraint_s_0: a_time_1_layer_0[0] = 0 ->
   s_time_1_layer_0[0] <= 0
 state_time_step_1indicator_constraint_z_1: a_time_1_layer_1[0] = 1 ->
   z_time_1_layer_1[0] <= 0
 state_time_step_1indicator_constraint_s_1: a_time_1_layer_1[0] = 0 ->
   s_time_1_layer_1[0] <= 0
 state_time_step_2indicator_constraint_z_0: a_time_2_layer_0[0] = 1 ->
   z_time_2_layer_0[0] <= 0
 state_time_step_2indicator_constraint_s_0: a_time_2_layer_0[0] = 0 ->
   s_time_2_layer_0[0] <= 0
 state_time_step_2indicator_constraint_z_1: a_time_2_layer_1[0] = 1 ->
   z_time_2_layer_1[0] <= 0
 state_time_step_2indicator_constraint_s_1: a_time_2_layer_1[0] = 0 ->
   s_time_2_layer_1[0] <= 0
 state_time_step_3indicator_constraint_z_0: a_time_3_layer_0[0] = 1 ->
   z_time_3_layer_0[0] <= 0
 state_time_step_3indicator_constraint_s_0: a_time_3_layer_0[0] = 0 ->
   s_time_3_layer_0[0] <= 0
 state_time_step_3indicator_constraint_z_1: a_time_3_layer_1[0] = 1 ->
   z_time_3_layer_1[0] <= 0
 state_time_step_3indicator_constraint_s_1: a_time_3_layer_1[0] = 0 ->
   s_time_3_layer_1[0] <= 0
 state_time_step_4indicator_constraint_z_0: a_time_4_layer_0[0] = 1 ->
   z_time_4_layer_0[0] <= 0
 state_time_step_4indicator_constraint_s_0: a_time_4_layer_0[0] = 0 ->
   s_time_4_layer_0[0] <= 0
 state_time_step_4indicator_constraint_z_1: a_time_4_layer_1[0] = 1 ->
   z_time_4_layer_1[0] <= 0
 state_time_step_4indicator_constraint_s_1: a_time_4_layer_1[0] = 0 ->
   s_time_4_layer_1[0] <= 0
 state_time_step_5indicator_constraint_z_0: a_time_5_layer_0[0] = 1 ->
   z_time_5_layer_0[0] <= 0
 state_time_step_5indicator_constraint_s_0: a_time_5_layer_0[0] = 0 ->
   s_time_5_layer_0[0] <= 0
 state_time_step_5indicator_constraint_z_1: a_time_5_layer_1[0] = 1 ->
   z_time_5_layer_1[0] <= 0
 state_time_step_5indicator_constraint_s_1: a_time_5_layer_1[0] = 0 ->
   s_time_5_layer_1[0] <= 0
 state_time_step_6indicator_constraint_z_0: a_time_6_layer_0[0] = 1 ->
   z_time_6_layer_0[0] <= 0
 state_time_step_6indicator_constraint_s_0: a_time_6_layer_0[0] = 0 ->
   s_time_6_layer_0[0] <= 0
 state_time_step_6indicator_constraint_z_1: a_time_6_layer_1[0] = 1 ->
   z_time_6_layer_1[0] <= 0
 state_time_step_6indicator_constraint_s_1: a_time_6_layer_1[0] = 0 ->
   s_time_6_layer_1[0] <= 0
 state_time_step_7indicator_constraint_z_0: a_time_7_layer_0[0] = 1 ->
   z_time_7_layer_0[0] <= 0
 state_time_step_7indicator_constraint_s_0: a_time_7_layer_0[0] = 0 ->
   s_time_7_layer_0[0] <= 0
 state_time_step_7indicator_constraint_z_1: a_time_7_layer_1[0] = 1 ->
   z_time_7_layer_1[0] <= 0
 state_time_step_7indicator_constraint_s_1: a_time_7_layer_1[0] = 0 ->
   s_time_7_layer_1[0] <= 0
 state_time_step_8indicator_constraint_z_0: a_time_8_layer_0[0] = 1 ->
   z_time_8_layer_0[0] <= 0
 state_time_step_8indicator_constraint_s_0: a_time_8_layer_0[0] = 0 ->
   s_time_8_layer_0[0] <= 0
 state_time_step_8indicator_constraint_z_1: a_time_8_layer_1[0] = 1 ->
   z_time_8_layer_1[0] <= 0
 state_time_step_8indicator_constraint_s_1: a_time_8_layer_1[0] = 0 ->
   s_time_8_layer_1[0] <= 0
 state_time_step_9indicator_constraint_z_0: a_time_9_layer_0[0] = 1 ->
   z_time_9_layer_0[0] <= 0
 state_time_step_9indicator_constraint_s_0: a_time_9_layer_0[0] = 0 ->
   s_time_9_layer_0[0] <= 0
 state_time_step_9indicator_constraint_z_1: a_time_9_layer_1[0] = 1 ->
   z_time_9_layer_1[0] <= 0
 state_time_step_9indicator_constraint_s_1: a_time_9_layer_1[0] = 0 ->
   s_time_9_layer_1[0] <= 0
 state_time_step_10indicator_constraint_z_0: a_time_10_layer_0[0] = 1 ->
   z_time_10_layer_0[0] <= 0
 state_time_step_10indicator_constraint_s_0: a_time_10_layer_0[0] = 0 ->
   s_time_10_layer_0[0] <= 0
 state_time_step_10indicator_constraint_z_1: a_time_10_layer_1[0] = 1 ->
   z_time_10_layer_1[0] <= 0
 state_time_step_10indicator_constraint_s_1: a_time_10_layer_1[0] = 0 ->
   s_time_10_layer_1[0] <= 0
 state_time_step_11indicator_constraint_z_0: a_time_11_layer_0[0] = 1 ->
   z_time_11_layer_0[0] <= 0
 state_time_step_11indicator_constraint_s_0: a_time_11_layer_0[0] = 0 ->
   s_time_11_layer_0[0] <= 0
 state_time_step_11indicator_constraint_z_1: a_time_11_layer_1[0] = 1 ->
   z_time_11_layer_1[0] <= 0
 state_time_step_11indicator_constraint_s_1: a_time_11_layer_1[0] = 0 ->
   s_time_11_layer_1[0] <= 0
 output_time_step_0indicator_constraint_z_0: a_time_0_layer_0[0] = 1 ->
   z_time_0_layer_0[0] <= 0
 output_time_step_0indicator_constraint_s_0: a_time_0_layer_0[0] = 0 ->
   s_time_0_layer_0[0] <= 0
 output_time_step_0indicator_constraint_z_1: a_time_0_layer_1[0] = 1 ->
   z_time_0_layer_1[0] <= 0
 output_time_step_0indicator_constraint_s_1: a_time_0_layer_1[0] = 0 ->
   s_time_0_layer_1[0] <= 0
 output_time_step_1indicator_constraint_z_0: a_time_1_layer_0[0] = 1 ->
   z_time_1_layer_0[0] <= 0
 output_time_step_1indicator_constraint_s_0: a_time_1_layer_0[0] = 0 ->
   s_time_1_layer_0[0] <= 0
 output_time_step_1indicator_constraint_z_1: a_time_1_layer_1[0] = 1 ->
   z_time_1_layer_1[0] <= 0
 output_time_step_1indicator_constraint_s_1: a_time_1_layer_1[0] = 0 ->
   s_time_1_layer_1[0] <= 0
 output_time_step_2indicator_constraint_z_0: a_time_2_layer_0[0] = 1 ->
   z_time_2_layer_0[0] <= 0
 output_time_step_2indicator_constraint_s_0: a_time_2_layer_0[0] = 0 ->
   s_time_2_layer_0[0] <= 0
 output_time_step_2indicator_constraint_z_1: a_time_2_layer_1[0] = 1 ->
   z_time_2_layer_1[0] <= 0
 output_time_step_2indicator_constraint_s_1: a_time_2_layer_1[0] = 0 ->
   s_time_2_layer_1[0] <= 0
 output_time_step_3indicator_constraint_z_0: a_time_3_layer_0[0] = 1 ->
   z_time_3_layer_0[0] <= 0
 output_time_step_3indicator_constraint_s_0: a_time_3_layer_0[0] = 0 ->
   s_time_3_layer_0[0] <= 0
 output_time_step_3indicator_constraint_z_1: a_time_3_layer_1[0] = 1 ->
   z_time_3_layer_1[0] <= 0
 output_time_step_3indicator_constraint_s_1: a_time_3_layer_1[0] = 0 ->
   s_time_3_layer_1[0] <= 0
 output_time_step_4indicator_constraint_z_0: a_time_4_layer_0[0] = 1 ->
   z_time_4_layer_0[0] <= 0
 output_time_step_4indicator_constraint_s_0: a_time_4_layer_0[0] = 0 ->
   s_time_4_layer_0[0] <= 0
 output_time_step_4indicator_constraint_z_1: a_time_4_layer_1[0] = 1 ->
   z_time_4_layer_1[0] <= 0
 output_time_step_4indicator_constraint_s_1: a_time_4_layer_1[0] = 0 ->
   s_time_4_layer_1[0] <= 0
 output_time_step_5indicator_constraint_z_0: a_time_5_layer_0[0] = 1 ->
   z_time_5_layer_0[0] <= 0
 output_time_step_5indicator_constraint_s_0: a_time_5_layer_0[0] = 0 ->
   s_time_5_layer_0[0] <= 0
 output_time_step_5indicator_constraint_z_1: a_time_5_layer_1[0] = 1 ->
   z_time_5_layer_1[0] <= 0
 output_time_step_5indicator_constraint_s_1: a_time_5_layer_1[0] = 0 ->
   s_time_5_layer_1[0] <= 0
 output_time_step_6indicator_constraint_z_0: a_time_6_layer_0[0] = 1 ->
   z_time_6_layer_0[0] <= 0
 output_time_step_6indicator_constraint_s_0: a_time_6_layer_0[0] = 0 ->
   s_time_6_layer_0[0] <= 0
 output_time_step_6indicator_constraint_z_1: a_time_6_layer_1[0] = 1 ->
   z_time_6_layer_1[0] <= 0
 output_time_step_6indicator_constraint_s_1: a_time_6_layer_1[0] = 0 ->
   s_time_6_layer_1[0] <= 0
 output_time_step_7indicator_constraint_z_0: a_time_7_layer_0[0] = 1 ->
   z_time_7_layer_0[0] <= 0
 output_time_step_7indicator_constraint_s_0: a_time_7_layer_0[0] = 0 ->
   s_time_7_layer_0[0] <= 0
 output_time_step_7indicator_constraint_z_1: a_time_7_layer_1[0] = 1 ->
   z_time_7_layer_1[0] <= 0
 output_time_step_7indicator_constraint_s_1: a_time_7_layer_1[0] = 0 ->
   s_time_7_layer_1[0] <= 0
 output_time_step_8indicator_constraint_z_0: a_time_8_layer_0[0] = 1 ->
   z_time_8_layer_0[0] <= 0
 output_time_step_8indicator_constraint_s_0: a_time_8_layer_0[0] = 0 ->
   s_time_8_layer_0[0] <= 0
 output_time_step_8indicator_constraint_z_1: a_time_8_layer_1[0] = 1 ->
   z_time_8_layer_1[0] <= 0
 output_time_step_8indicator_constraint_s_1: a_time_8_layer_1[0] = 0 ->
   s_time_8_layer_1[0] <= 0
 output_time_step_9indicator_constraint_z_0: a_time_9_layer_0[0] = 1 ->
   z_time_9_layer_0[0] <= 0
 output_time_step_9indicator_constraint_s_0: a_time_9_layer_0[0] = 0 ->
   s_time_9_layer_0[0] <= 0
 output_time_step_9indicator_constraint_z_1: a_time_9_layer_1[0] = 1 ->
   z_time_9_layer_1[0] <= 0
 output_time_step_9indicator_constraint_s_1: a_time_9_layer_1[0] = 0 ->
   s_time_9_layer_1[0] <= 0
 output_time_step_10indicator_constraint_z_0: a_time_10_layer_0[0] = 1 ->
   z_time_10_layer_0[0] <= 0
 output_time_step_10indicator_constraint_s_0: a_time_10_layer_0[0] = 0 ->
   s_time_10_layer_0[0] <= 0
 output_time_step_10indicator_constraint_z_1: a_time_10_layer_1[0] = 1 ->
   z_time_10_layer_1[0] <= 0
 output_time_step_10indicator_constraint_s_1: a_time_10_layer_1[0] = 0 ->
   s_time_10_layer_1[0] <= 0
 output_time_step_11indicator_constraint_z_0: a_time_11_layer_0[0] = 1 ->
   z_time_11_layer_0[0] <= 0
 output_time_step_11indicator_constraint_s_0: a_time_11_layer_0[0] = 0 ->
   s_time_11_layer_0[0] <= 0
 output_time_step_11indicator_constraint_z_1: a_time_11_layer_1[0] = 1 ->
   z_time_11_layer_1[0] <= 0
 output_time_step_11indicator_constraint_s_1: a_time_11_layer_1[0] = 0 ->
   s_time_11_layer_1[0] <= 0
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
 a_time_0_layer_0[0] a_time_0_layer_1[0] a_time_1_layer_0[0]
 a_time_1_layer_1[0] a_time_2_layer_0[0] a_time_2_layer_1[0]
 a_time_3_layer_0[0] a_time_3_layer_1[0] a_time_4_layer_0[0]
 a_time_4_layer_1[0] a_time_5_layer_0[0] a_time_5_layer_1[0]
 a_time_6_layer_0[0] a_time_6_layer_1[0] a_time_7_layer_0[0]
 a_time_7_layer_1[0] a_time_8_layer_0[0] a_time_8_layer_1[0]
 a_time_9_layer_0[0] a_time_9_layer_1[0] a_time_10_layer_0[0]
 a_time_10_layer_1[0] a_time_11_layer_0[0] a_time_11_layer_1[0]
 a_time_0_layer_0[0] a_time_0_layer_1[0] a_time_1_layer_0[0]
 a_time_1_layer_1[0] a_time_2_layer_0[0] a_time_2_layer_1[0]
 a_time_3_layer_0[0] a_time_3_layer_1[0] a_time_4_layer_0[0]
 a_time_4_layer_1[0] a_time_5_layer_0[0] a_time_5_layer_1[0]
 a_time_6_layer_0[0] a_time_6_layer_1[0] a_time_7_layer_0[0]
 a_time_7_layer_1[0] a_time_8_layer_0[0] a_time_8_layer_1[0]
 a_time_9_layer_0[0] a_time_9_layer_1[0] a_time_10_layer_0[0]
 a_time_10_layer_1[0] a_time_11_layer_0[0] a_time_11_layer_1[0]
End
