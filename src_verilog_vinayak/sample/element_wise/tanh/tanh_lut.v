// Complete tanh LUT RAM module with all 276 values
// Range: 0.25 to 3.0 with 0.01 increment in S1.5.6 fixed-point format
module tanh_lut_ram #(
    parameter DATA_WIDTH = 12,           // 12-bit fixed-point S1.5.6 format
    parameter ADDR_WIDTH = 9,            // 9 bits for 276 addresses (0 to 275)
    parameter LUT_SIZE = 276             // Number of LUT entries
)(
    input  [ADDR_WIDTH-1:0] addr,    // Address input (0-275)
    output  reg [DATA_WIDTH-1:0] tanh_out // 12-bit tanh output
);

    // LUT RAM declaration
    reg [DATA_WIDTH-1:0] tanh_lut [0:LUT_SIZE-1];
    
    // Initialize LUT with ALL 276 tanh values
    // Complete table from 0.25 to 3.0 with 0.01 increment
    initial begin
        tanh_lut[  0] = 12'd15;   // tanh(0.25) ~= 0.244919
        tanh_lut[  1] = 12'd16;   // tanh(0.26) ~= 0.254296
        tanh_lut[  2] = 12'd16;   // tanh(0.27) ~= 0.263625
        tanh_lut[  3] = 12'd17;   // tanh(0.28) ~= 0.272905
        tanh_lut[  4] = 12'd18;   // tanh(0.29) ~= 0.282135
        tanh_lut[  5] = 12'd18;   // tanh(0.30) ~= 0.291313
        tanh_lut[  6] = 12'd19;   // tanh(0.31) ~= 0.300437
        tanh_lut[  7] = 12'd19;   // tanh(0.32) ~= 0.309507
        tanh_lut[  8] = 12'd20;   // tanh(0.33) ~= 0.318521
        tanh_lut[  9] = 12'd20;   // tanh(0.34) ~= 0.327477
        tanh_lut[ 10] = 12'd21;   // tanh(0.35) ~= 0.336376
        tanh_lut[ 11] = 12'd22;   // tanh(0.36) ~= 0.345214
        tanh_lut[ 12] = 12'd22;   // tanh(0.37) ~= 0.353992
        tanh_lut[ 13] = 12'd23;   // tanh(0.38) ~= 0.362707
        tanh_lut[ 14] = 12'd23;   // tanh(0.39) ~= 0.371360
        tanh_lut[ 15] = 12'd24;   // tanh(0.40) ~= 0.379949
        tanh_lut[ 16] = 12'd24;   // tanh(0.41) ~= 0.388473
        tanh_lut[ 17] = 12'd25;   // tanh(0.42) ~= 0.396930
        tanh_lut[ 18] = 12'd25;   // tanh(0.43) ~= 0.405321
        tanh_lut[ 19] = 12'd26;   // tanh(0.44) ~= 0.413644
        tanh_lut[ 20] = 12'd27;   // tanh(0.45) ~= 0.421899
        tanh_lut[ 21] = 12'd27;   // tanh(0.46) ~= 0.430084
        tanh_lut[ 22] = 12'd28;   // tanh(0.47) ~= 0.438199
        tanh_lut[ 23] = 12'd28;   // tanh(0.48) ~= 0.446244
        tanh_lut[ 24] = 12'd29;   // tanh(0.49) ~= 0.454216
        tanh_lut[ 25] = 12'd29;   // tanh(0.50) ~= 0.462117
        tanh_lut[ 26] = 12'd30;   // tanh(0.51) ~= 0.469945
        tanh_lut[ 27] = 12'd30;   // tanh(0.52) ~= 0.477700
        tanh_lut[ 28] = 12'd31;   // tanh(0.53) ~= 0.485381
        tanh_lut[ 29] = 12'd31;   // tanh(0.54) ~= 0.492988
        tanh_lut[ 30] = 12'd32;   // tanh(0.55) ~= 0.500520
        tanh_lut[ 31] = 12'd32;   // tanh(0.56) ~= 0.507977
        tanh_lut[ 32] = 12'd32;   // tanh(0.57) ~= 0.515359
        tanh_lut[ 33] = 12'd33;   // tanh(0.58) ~= 0.522665
        tanh_lut[ 34] = 12'd33;   // tanh(0.59) ~= 0.529896
        tanh_lut[ 35] = 12'd34;   // tanh(0.60) ~= 0.537050
        tanh_lut[ 36] = 12'd34;   // tanh(0.61) ~= 0.544127
        tanh_lut[ 37] = 12'd35;   // tanh(0.62) ~= 0.551128
        tanh_lut[ 38] = 12'd35;   // tanh(0.63) ~= 0.558052
        tanh_lut[ 39] = 12'd36;   // tanh(0.64) ~= 0.564900
        tanh_lut[ 40] = 12'd36;   // tanh(0.65) ~= 0.571670
        tanh_lut[ 41] = 12'd37;   // tanh(0.66) ~= 0.578363
        tanh_lut[ 42] = 12'd37;   // tanh(0.67) ~= 0.584980
        tanh_lut[ 43] = 12'd37;   // tanh(0.68) ~= 0.591519
        tanh_lut[ 44] = 12'd38;   // tanh(0.69) ~= 0.597982
        tanh_lut[ 45] = 12'd38;   // tanh(0.70) ~= 0.604368
        tanh_lut[ 46] = 12'd39;   // tanh(0.71) ~= 0.610677
        tanh_lut[ 47] = 12'd39;   // tanh(0.72) ~= 0.616909
        tanh_lut[ 48] = 12'd39;   // tanh(0.73) ~= 0.623065
        tanh_lut[ 49] = 12'd40;   // tanh(0.74) ~= 0.629145
        tanh_lut[ 50] = 12'd40;   // tanh(0.75) ~= 0.635149
        tanh_lut[ 51] = 12'd41;   // tanh(0.76) ~= 0.641077
        tanh_lut[ 52] = 12'd41;   // tanh(0.77) ~= 0.646929
        tanh_lut[ 53] = 12'd41;   // tanh(0.78) ~= 0.652707
        tanh_lut[ 54] = 12'd42;   // tanh(0.79) ~= 0.658409
        tanh_lut[ 55] = 12'd42;   // tanh(0.80) ~= 0.664037
        tanh_lut[ 56] = 12'd42;   // tanh(0.81) ~= 0.669590
        tanh_lut[ 57] = 12'd43;   // tanh(0.82) ~= 0.675070
        tanh_lut[ 58] = 12'd43;   // tanh(0.83) ~= 0.680476
        tanh_lut[ 59] = 12'd43;   // tanh(0.84) ~= 0.685809
        tanh_lut[ 60] = 12'd44;   // tanh(0.85) ~= 0.691069
        tanh_lut[ 61] = 12'd44;   // tanh(0.86) ~= 0.696258
        tanh_lut[ 62] = 12'd44;   // tanh(0.87) ~= 0.701374
        tanh_lut[ 63] = 12'd45;   // tanh(0.88) ~= 0.706419
        tanh_lut[ 64] = 12'd45;   // tanh(0.89) ~= 0.711394
        tanh_lut[ 65] = 12'd45;   // tanh(0.90) ~= 0.716298
        tanh_lut[ 66] = 12'd46;   // tanh(0.91) ~= 0.721132
        tanh_lut[ 67] = 12'd46;   // tanh(0.92) ~= 0.725897
        tanh_lut[ 68] = 12'd46;   // tanh(0.93) ~= 0.730594
        tanh_lut[ 69] = 12'd47;   // tanh(0.94) ~= 0.735222
        tanh_lut[ 70] = 12'd47;   // tanh(0.95) ~= 0.739783
        tanh_lut[ 71] = 12'd47;   // tanh(0.96) ~= 0.744277
        tanh_lut[ 72] = 12'd47;   // tanh(0.97) ~= 0.748704
        tanh_lut[ 73] = 12'd48;   // tanh(0.98) ~= 0.753066
        tanh_lut[ 74] = 12'd48;   // tanh(0.99) ~= 0.757362
        tanh_lut[ 75] = 12'd48;   // tanh(1.00) ~= 0.761594
        tanh_lut[ 76] = 12'd49;   // tanh(1.01) ~= 0.765762
        tanh_lut[ 77] = 12'd49;   // tanh(1.02) ~= 0.769867
        tanh_lut[ 78] = 12'd49;   // tanh(1.03) ~= 0.773908
        tanh_lut[ 79] = 12'd49;   // tanh(1.04) ~= 0.777888
        tanh_lut[ 80] = 12'd50;   // tanh(1.05) ~= 0.781806
        tanh_lut[ 81] = 12'd50;   // tanh(1.06) ~= 0.785664
        tanh_lut[ 82] = 12'd50;   // tanh(1.07) ~= 0.789461
        tanh_lut[ 83] = 12'd50;   // tanh(1.08) ~= 0.793199
        tanh_lut[ 84] = 12'd51;   // tanh(1.09) ~= 0.796878
        tanh_lut[ 85] = 12'd51;   // tanh(1.10) ~= 0.800499
        tanh_lut[ 86] = 12'd51;   // tanh(1.11) ~= 0.804062
        tanh_lut[ 87] = 12'd51;   // tanh(1.12) ~= 0.807569
        tanh_lut[ 88] = 12'd51;   // tanh(1.13) ~= 0.811019
        tanh_lut[ 89] = 12'd52;   // tanh(1.14) ~= 0.814414
        tanh_lut[ 90] = 12'd52;   // tanh(1.15) ~= 0.817754
        tanh_lut[ 91] = 12'd52;   // tanh(1.16) ~= 0.821040
        tanh_lut[ 92] = 12'd52;   // tanh(1.17) ~= 0.824272
        tanh_lut[ 93] = 12'd52;   // tanh(1.18) ~= 0.827452
        tanh_lut[ 94] = 12'd53;   // tanh(1.19) ~= 0.830579
        tanh_lut[ 95] = 12'd53;   // tanh(1.20) ~= 0.833655
        tanh_lut[ 96] = 12'd53;   // tanh(1.21) ~= 0.836679
        tanh_lut[ 97] = 12'd53;   // tanh(1.22) ~= 0.839654
        tanh_lut[ 98] = 12'd53;   // tanh(1.23) ~= 0.842579
        tanh_lut[ 99] = 12'd54;   // tanh(1.24) ~= 0.845456
        tanh_lut[100] = 12'd54;   // tanh(1.25) ~= 0.848284
        tanh_lut[101] = 12'd54;   // tanh(1.26) ~= 0.851064
        tanh_lut[102] = 12'd54;   // tanh(1.27) ~= 0.853798
        tanh_lut[103] = 12'd54;   // tanh(1.28) ~= 0.856485
        tanh_lut[104] = 12'd54;   // tanh(1.29) ~= 0.859127
        tanh_lut[105] = 12'd55;   // tanh(1.30) ~= 0.861723
        tanh_lut[106] = 12'd55;   // tanh(1.31) ~= 0.864275
        tanh_lut[107] = 12'd55;   // tanh(1.32) ~= 0.866784
        tanh_lut[108] = 12'd55;   // tanh(1.33) ~= 0.869249
        tanh_lut[109] = 12'd55;   // tanh(1.34) ~= 0.871672
        tanh_lut[110] = 12'd55;   // tanh(1.35) ~= 0.874053
        tanh_lut[111] = 12'd56;   // tanh(1.36) ~= 0.876393
        tanh_lut[112] = 12'd56;   // tanh(1.37) ~= 0.878692
        tanh_lut[113] = 12'd56;   // tanh(1.38) ~= 0.880951
        tanh_lut[114] = 12'd56;   // tanh(1.39) ~= 0.883171
        tanh_lut[115] = 12'd56;   // tanh(1.40) ~= 0.885352
        tanh_lut[116] = 12'd56;   // tanh(1.41) ~= 0.887494
        tanh_lut[117] = 12'd56;   // tanh(1.42) ~= 0.889599
        tanh_lut[118] = 12'd57;   // tanh(1.43) ~= 0.891667
        tanh_lut[119] = 12'd57;   // tanh(1.44) ~= 0.893698
        tanh_lut[120] = 12'd57;   // tanh(1.45) ~= 0.895693
        tanh_lut[121] = 12'd57;   // tanh(1.46) ~= 0.897653
        tanh_lut[122] = 12'd57;   // tanh(1.47) ~= 0.899577
        tanh_lut[123] = 12'd57;   // tanh(1.48) ~= 0.901468
        tanh_lut[124] = 12'd57;   // tanh(1.49) ~= 0.903325
        tanh_lut[125] = 12'd57;   // tanh(1.50) ~= 0.905148
        tanh_lut[126] = 12'd58;   // tanh(1.51) ~= 0.906939
        tanh_lut[127] = 12'd58;   // tanh(1.52) ~= 0.908698
        tanh_lut[128] = 12'd58;   // tanh(1.53) ~= 0.910425
        tanh_lut[129] = 12'd58;   // tanh(1.54) ~= 0.912120
        tanh_lut[130] = 12'd58;   // tanh(1.55) ~= 0.913785
        tanh_lut[131] = 12'd58;   // tanh(1.56) ~= 0.915420
        tanh_lut[132] = 12'd58;   // tanh(1.57) ~= 0.917026
        tanh_lut[133] = 12'd58;   // tanh(1.58) ~= 0.918602
        tanh_lut[134] = 12'd58;   // tanh(1.59) ~= 0.920149
        tanh_lut[135] = 12'd58;   // tanh(1.60) ~= 0.921669
        tanh_lut[136] = 12'd59;   // tanh(1.61) ~= 0.923160
        tanh_lut[137] = 12'd59;   // tanh(1.62) ~= 0.924624
        tanh_lut[138] = 12'd59;   // tanh(1.63) ~= 0.926062
        tanh_lut[139] = 12'd59;   // tanh(1.64) ~= 0.927473
        tanh_lut[140] = 12'd59;   // tanh(1.65) ~= 0.928858
        tanh_lut[141] = 12'd59;   // tanh(1.66) ~= 0.930217
        tanh_lut[142] = 12'd59;   // tanh(1.67) ~= 0.931552
        tanh_lut[143] = 12'd59;   // tanh(1.68) ~= 0.932862
        tanh_lut[144] = 12'd59;   // tanh(1.69) ~= 0.934147
        tanh_lut[145] = 12'd59;   // tanh(1.70) ~= 0.935409
        tanh_lut[146] = 12'd59;   // tanh(1.71) ~= 0.936648
        tanh_lut[147] = 12'd60;   // tanh(1.72) ~= 0.937863
        tanh_lut[148] = 12'd60;   // tanh(1.73) ~= 0.939056
        tanh_lut[149] = 12'd60;   // tanh(1.74) ~= 0.940227
        tanh_lut[150] = 12'd60;   // tanh(1.75) ~= 0.941376
        tanh_lut[151] = 12'd60;   // tanh(1.76) ~= 0.942503
        tanh_lut[152] = 12'd60;   // tanh(1.77) ~= 0.943609
        tanh_lut[153] = 12'd60;   // tanh(1.78) ~= 0.944695
        tanh_lut[154] = 12'd60;   // tanh(1.79) ~= 0.945761
        tanh_lut[155] = 12'd60;   // tanh(1.80) ~= 0.946806
        tanh_lut[156] = 12'd60;   // tanh(1.81) ~= 0.947832
        tanh_lut[157] = 12'd60;   // tanh(1.82) ~= 0.948838
        tanh_lut[158] = 12'd60;   // tanh(1.83) ~= 0.949826
        tanh_lut[159] = 12'd60;   // tanh(1.84) ~= 0.950795
        tanh_lut[160] = 12'd60;   // tanh(1.85) ~= 0.951746
        tanh_lut[161] = 12'd60;   // tanh(1.86) ~= 0.952679
        tanh_lut[162] = 12'd61;   // tanh(1.87) ~= 0.953594
        tanh_lut[163] = 12'd61;   // tanh(1.88) ~= 0.954492
        tanh_lut[164] = 12'd61;   // tanh(1.89) ~= 0.955373
        tanh_lut[165] = 12'd61;   // tanh(1.90) ~= 0.956237
        tanh_lut[166] = 12'd61;   // tanh(1.91) ~= 0.957085
        tanh_lut[167] = 12'd61;   // tanh(1.92) ~= 0.957917
        tanh_lut[168] = 12'd61;   // tanh(1.93) ~= 0.958733
        tanh_lut[169] = 12'd61;   // tanh(1.94) ~= 0.959534
        tanh_lut[170] = 12'd61;   // tanh(1.95) ~= 0.960319
        tanh_lut[171] = 12'd61;   // tanh(1.96) ~= 0.961090
        tanh_lut[172] = 12'd61;   // tanh(1.97) ~= 0.961846
        tanh_lut[173] = 12'd61;   // tanh(1.98) ~= 0.962587
        tanh_lut[174] = 12'd61;   // tanh(1.99) ~= 0.963314
        tanh_lut[175] = 12'd61;   // tanh(2.00) ~= 0.964028
        tanh_lut[176] = 12'd61;   // tanh(2.01) ~= 0.964727
        tanh_lut[177] = 12'd61;   // tanh(2.02) ~= 0.965414
        tanh_lut[178] = 12'd61;   // tanh(2.03) ~= 0.966087
        tanh_lut[179] = 12'd61;   // tanh(2.04) ~= 0.966747
        tanh_lut[180] = 12'd61;   // tanh(2.05) ~= 0.967395
        tanh_lut[181] = 12'd61;   // tanh(2.06) ~= 0.968030
        tanh_lut[182] = 12'd61;   // tanh(2.07) ~= 0.968653
        tanh_lut[183] = 12'd62;   // tanh(2.08) ~= 0.969265
        tanh_lut[184] = 12'd62;   // tanh(2.09) ~= 0.969864
        tanh_lut[185] = 12'd62;   // tanh(2.10) ~= 0.970452
        tanh_lut[186] = 12'd62;   // tanh(2.11) ~= 0.971029
        tanh_lut[187] = 12'd62;   // tanh(2.12) ~= 0.971594
        tanh_lut[188] = 12'd62;   // tanh(2.13) ~= 0.972149
        tanh_lut[189] = 12'd62;   // tanh(2.14) ~= 0.972693
        tanh_lut[190] = 12'd62;   // tanh(2.15) ~= 0.973226
        tanh_lut[191] = 12'd62;   // tanh(2.16) ~= 0.973749
        tanh_lut[192] = 12'd62;   // tanh(2.17) ~= 0.974262
        tanh_lut[193] = 12'd62;   // tanh(2.18) ~= 0.974766
        tanh_lut[194] = 12'd62;   // tanh(2.19) ~= 0.975259
        tanh_lut[195] = 12'd62;   // tanh(2.20) ~= 0.975743
        tanh_lut[196] = 12'd62;   // tanh(2.21) ~= 0.976218
        tanh_lut[197] = 12'd62;   // tanh(2.22) ~= 0.976683
        tanh_lut[198] = 12'd62;   // tanh(2.23) ~= 0.977140
        tanh_lut[199] = 12'd62;   // tanh(2.24) ~= 0.977587
        tanh_lut[200] = 12'd62;   // tanh(2.25) ~= 0.978026
        tanh_lut[201] = 12'd62;   // tanh(2.26) ~= 0.978457
        tanh_lut[202] = 12'd62;   // tanh(2.27) ~= 0.978879
        tanh_lut[203] = 12'd62;   // tanh(2.28) ~= 0.979293
        tanh_lut[204] = 12'd62;   // tanh(2.29) ~= 0.979698
        tanh_lut[205] = 12'd62;   // tanh(2.30) ~= 0.980096
        tanh_lut[206] = 12'd62;   // tanh(2.31) ~= 0.980487
        tanh_lut[207] = 12'd62;   // tanh(2.32) ~= 0.980869
        tanh_lut[208] = 12'd62;   // tanh(2.33) ~= 0.981245
        tanh_lut[209] = 12'd62;   // tanh(2.34) ~= 0.981613
        tanh_lut[210] = 12'd62;   // tanh(2.35) ~= 0.981973
        tanh_lut[211] = 12'd62;   // tanh(2.36) ~= 0.982327
        tanh_lut[212] = 12'd62;   // tanh(2.37) ~= 0.982674
        tanh_lut[213] = 12'd62;   // tanh(2.38) ~= 0.983014
        tanh_lut[214] = 12'd62;   // tanh(2.39) ~= 0.983348
        tanh_lut[215] = 12'd62;   // tanh(2.40) ~= 0.983675
        tanh_lut[216] = 12'd62;   // tanh(2.41) ~= 0.983996
        tanh_lut[217] = 12'd62;   // tanh(2.42) ~= 0.984310
        tanh_lut[218] = 12'd63;   // tanh(2.43) ~= 0.984618
        tanh_lut[219] = 12'd63;   // tanh(2.44) ~= 0.984921
        tanh_lut[220] = 12'd63;   // tanh(2.45) ~= 0.985217
        tanh_lut[221] = 12'd63;   // tanh(2.46) ~= 0.985508
        tanh_lut[222] = 12'd63;   // tanh(2.47) ~= 0.985792
        tanh_lut[223] = 12'd63;   // tanh(2.48) ~= 0.986072
        tanh_lut[224] = 12'd63;   // tanh(2.49) ~= 0.986346
        tanh_lut[225] = 12'd63;   // tanh(2.50) ~= 0.986614
        tanh_lut[226] = 12'd63;   // tanh(2.51) ~= 0.986878
        tanh_lut[227] = 12'd63;   // tanh(2.52) ~= 0.987136
        tanh_lut[228] = 12'd63;   // tanh(2.53) ~= 0.987389
        tanh_lut[229] = 12'd63;   // tanh(2.54) ~= 0.987637
        tanh_lut[230] = 12'd63;   // tanh(2.55) ~= 0.987880
        tanh_lut[231] = 12'd63;   // tanh(2.56) ~= 0.988119
        tanh_lut[232] = 12'd63;   // tanh(2.57) ~= 0.988353
        tanh_lut[233] = 12'd63;   // tanh(2.58) ~= 0.988582
        tanh_lut[234] = 12'd63;   // tanh(2.59) ~= 0.988807
        tanh_lut[235] = 12'd63;   // tanh(2.60) ~= 0.989027
        tanh_lut[236] = 12'd63;   // tanh(2.61) ~= 0.989244
        tanh_lut[237] = 12'd63;   // tanh(2.62) ~= 0.989455
        tanh_lut[238] = 12'd63;   // tanh(2.63) ~= 0.989663
        tanh_lut[239] = 12'd63;   // tanh(2.64) ~= 0.989867
        tanh_lut[240] = 12'd63;   // tanh(2.65) ~= 0.990066
        tanh_lut[241] = 12'd63;   // tanh(2.66) ~= 0.990262
        tanh_lut[242] = 12'd63;   // tanh(2.67) ~= 0.990454
        tanh_lut[243] = 12'd63;   // tanh(2.68) ~= 0.990642
        tanh_lut[244] = 12'd63;   // tanh(2.69) ~= 0.990827
        tanh_lut[245] = 12'd63;   // tanh(2.70) ~= 0.991007
        tanh_lut[246] = 12'd63;   // tanh(2.71) ~= 0.991185
        tanh_lut[247] = 12'd63;   // tanh(2.72) ~= 0.991359
        tanh_lut[248] = 12'd63;   // tanh(2.73) ~= 0.991529
        tanh_lut[249] = 12'd63;   // tanh(2.74) ~= 0.991696
        tanh_lut[250] = 12'd63;   // tanh(2.75) ~= 0.991860
        tanh_lut[251] = 12'd63;   // tanh(2.76) ~= 0.992020
        tanh_lut[252] = 12'd63;   // tanh(2.77) ~= 0.992178
        tanh_lut[253] = 12'd63;   // tanh(2.78) ~= 0.992332
        tanh_lut[254] = 12'd63;   // tanh(2.79) ~= 0.992483
        tanh_lut[255] = 12'd63;   // tanh(2.80) ~= 0.992632
        tanh_lut[256] = 12'd63;   // tanh(2.81) ~= 0.992777
        tanh_lut[257] = 12'd63;   // tanh(2.82) ~= 0.992919
        tanh_lut[258] = 12'd63;   // tanh(2.83) ~= 0.993059
        tanh_lut[259] = 12'd63;   // tanh(2.84) ~= 0.993196
        tanh_lut[260] = 12'd63;   // tanh(2.85) ~= 0.993330
        tanh_lut[261] = 12'd63;   // tanh(2.86) ~= 0.993462
        tanh_lut[262] = 12'd63;   // tanh(2.87) ~= 0.993591
        tanh_lut[263] = 12'd63;   // tanh(2.88) ~= 0.993718
        tanh_lut[264] = 12'd63;   // tanh(2.89) ~= 0.993842
        tanh_lut[265] = 12'd63;   // tanh(2.90) ~= 0.993963
        tanh_lut[266] = 12'd63;   // tanh(2.91) ~= 0.994082
        tanh_lut[267] = 12'd63;   // tanh(2.92) ~= 0.994199
        tanh_lut[268] = 12'd63;   // tanh(2.93) ~= 0.994314
        tanh_lut[269] = 12'd63;   // tanh(2.94) ~= 0.994426
        tanh_lut[270] = 12'd63;   // tanh(2.95) ~= 0.994536
        tanh_lut[271] = 12'd63;   // tanh(2.96) ~= 0.994644
        tanh_lut[272] = 12'd63;   // tanh(2.97) ~= 0.994750
        tanh_lut[273] = 12'd63;   // tanh(2.98) ~= 0.994853
        tanh_lut[274] = 12'd63;   // tanh(2.99) ~= 0.994955
        tanh_lut[275] = 12'd63;   // tanh(3.00) ~= 0.995055
    end
    
    // RAM read operation
    always @(*) begin
        tanh_out <= tanh_lut[addr];
    end
endmodule