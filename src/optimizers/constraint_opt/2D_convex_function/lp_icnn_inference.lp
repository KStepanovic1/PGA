\ Model lp_icnn_inference
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  z_1[0]
Subject To
 I_layer_0[0]: 0.5366615653038025 x + z_0[0] >= 0
 I_layer_0[1]: - 1.220082879066467 x + z_0[1] >= -0.6115313172340393
 II_layer_0[0]: z_0[0] >= 0
 II_layer_0[1]: z_0[1] >= 0
 I_layer_1[0]: 0.4906525015830994 x - 0.8447712659835815 z_0[0]
   - 0.8168848752975464 z_0[1] + z_1[0] = 0.2063517272472382
Bounds
 x <= 1
 z_0[0] >= -1000
 z_0[1] >= -1000
 z_1[0] >= -1000
End
