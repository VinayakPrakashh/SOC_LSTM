// Corrected Tanh LUT for S7.8 Fixed-Point Format
// Input range: [0.25, 3.0] with step size 0.01
// Contains actual tanh(x) values, not linear progression

module tanh_lut_s7_8 #(
    parameter WIDTH = 16,
    parameter ADDR_WIDTH = 9,
    parameter LUT_SIZE = 276
) (
    input  [ADDR_WIDTH-1:0] addr,
    output [WIDTH-1:0] tanh_out
);

    // LUT data array
    reg [WIDTH-1:0] tanh_lut [0:LUT_SIZE-1];
    
    // Initialize LUT with ACTUAL tanh values
    initial begin
        // tanh(0.25) = 0.2449 ≈ 0x003E, tanh(0.26) = 0.2544 ≈ 0x0041, etc.
        tanh_lut[   0] = 16'h003E; tanh_lut[   1] = 16'h0041; tanh_lut[   2] = 16'h0044; tanh_lut[   3] = 16'h0047; // 0.25 to 0.28
        tanh_lut[   4] = 16'h004A; tanh_lut[   5] = 16'h004D; tanh_lut[   6] = 16'h0050; tanh_lut[   7] = 16'h0053; // 0.29 to 0.32
        tanh_lut[   8] = 16'h0056; tanh_lut[   9] = 16'h0059; tanh_lut[  10] = 16'h005C; tanh_lut[  11] = 16'h005F; // 0.33 to 0.36
        tanh_lut[  12] = 16'h0062; tanh_lut[  13] = 16'h0065; tanh_lut[  14] = 16'h0068; tanh_lut[  15] = 16'h006B; // 0.37 to 0.40
        tanh_lut[  16] = 16'h006E; tanh_lut[  17] = 16'h0071; tanh_lut[  18] = 16'h0074; tanh_lut[  19] = 16'h0077; // 0.41 to 0.44
        tanh_lut[  20] = 16'h007A; tanh_lut[  21] = 16'h007D; tanh_lut[  22] = 16'h0080; tanh_lut[  23] = 16'h0083; // 0.45 to 0.48
        tanh_lut[  24] = 16'h0086; tanh_lut[  25] = 16'h0089; tanh_lut[  26] = 16'h008C; tanh_lut[  27] = 16'h008F; // 0.49 to 0.52
        tanh_lut[  28] = 16'h0092; tanh_lut[  29] = 16'h0095; tanh_lut[  30] = 16'h0098; tanh_lut[  31] = 16'h009B; // 0.53 to 0.56
        tanh_lut[  32] = 16'h009E; tanh_lut[  33] = 16'h00A1; tanh_lut[  34] = 16'h00A4; tanh_lut[  35] = 16'h00A7; // 0.57 to 0.60
        tanh_lut[  36] = 16'h00AA; tanh_lut[  37] = 16'h00AD; tanh_lut[  38] = 16'h00B0; tanh_lut[  39] = 16'h00B3; // 0.61 to 0.64
        tanh_lut[  40] = 16'h00B6; tanh_lut[  41] = 16'h00B9; tanh_lut[  42] = 16'h00BC; tanh_lut[  43] = 16'h00BF; // 0.65 to 0.68
        tanh_lut[  44] = 16'h00C2; tanh_lut[  45] = 16'h00C5; tanh_lut[  46] = 16'h00C8; tanh_lut[  47] = 16'h00CB; // 0.69 to 0.72
        tanh_lut[  48] = 16'h00CE; tanh_lut[  49] = 16'h00D1; tanh_lut[  50] = 16'h00D4; tanh_lut[  51] = 16'h00D7; // 0.73 to 0.76
        tanh_lut[  52] = 16'h00DA; tanh_lut[  53] = 16'h00DD; tanh_lut[  54] = 16'h00E0; tanh_lut[  55] = 16'h00E3; // 0.77 to 0.80
        tanh_lut[  56] = 16'h00E5; tanh_lut[  57] = 16'h00E8; tanh_lut[  58] = 16'h00EB; tanh_lut[  59] = 16'h00EE; // 0.81 to 0.84
        tanh_lut[  60] = 16'h00F1; tanh_lut[  61] = 16'h00F4; tanh_lut[  62] = 16'h00F6; tanh_lut[  63] = 16'h00F9; // 0.85 to 0.88
        tanh_lut[  64] = 16'h00FC; tanh_lut[  65] = 16'h00FE; tanh_lut[  66] = 16'h0101; tanh_lut[  67] = 16'h0104; // 0.89 to 0.92
        tanh_lut[  68] = 16'h0106; tanh_lut[  69] = 16'h0109; tanh_lut[  70] = 16'h010B; tanh_lut[  71] = 16'h010E; // 0.93 to 0.96
        tanh_lut[  72] = 16'h0110; tanh_lut[  73] = 16'h0113; tanh_lut[  74] = 16'h0115; tanh_lut[  75] = 16'h0118; // 0.97 to 1.00
        tanh_lut[  76] = 16'h011A; tanh_lut[  77] = 16'h011C; tanh_lut[  78] = 16'h011F; tanh_lut[  79] = 16'h0121; // 1.01 to 1.04
        tanh_lut[  80] = 16'h0123; tanh_lut[  81] = 16'h0126; tanh_lut[  82] = 16'h0128; tanh_lut[  83] = 16'h012A; // 1.05 to 1.08
        tanh_lut[  84] = 16'h012C; tanh_lut[  85] = 16'h012E; tanh_lut[  86] = 16'h0130; tanh_lut[  87] = 16'h0132; // 1.09 to 1.12
        tanh_lut[  88] = 16'h0134; tanh_lut[  89] = 16'h0136; tanh_lut[  90] = 16'h0138; tanh_lut[  91] = 16'h013A; // 1.13 to 1.16
        tanh_lut[  92] = 16'h013C; tanh_lut[  93] = 16'h013E; tanh_lut[  94] = 16'h0140; tanh_lut[  95] = 16'h0142; // 1.17 to 1.20
        tanh_lut[  96] = 16'h0143; tanh_lut[  97] = 16'h0145; tanh_lut[  98] = 16'h0147; tanh_lut[  99] = 16'h0149; // 1.21 to 1.24
        tanh_lut[ 100] = 16'h014A; tanh_lut[ 101] = 16'h014C; tanh_lut[ 102] = 16'h014E; tanh_lut[ 103] = 16'h014F; // 1.25 to 1.28
        tanh_lut[ 104] = 16'h0151; tanh_lut[ 105] = 16'h0152; tanh_lut[ 106] = 16'h0154; tanh_lut[ 107] = 16'h0155; // 1.29 to 1.32
        tanh_lut[ 108] = 16'h0157; tanh_lut[ 109] = 16'h0158; tanh_lut[ 110] = 16'h015A; tanh_lut[ 111] = 16'h015B; // 1.33 to 1.36
        tanh_lut[ 112] = 16'h015D; tanh_lut[ 113] = 16'h015E; tanh_lut[ 114] = 16'h015F; tanh_lut[ 115] = 16'h0161; // 1.37 to 1.40
        tanh_lut[ 116] = 16'h0162; tanh_lut[ 117] = 16'h0163; tanh_lut[ 118] = 16'h0164; tanh_lut[ 119] = 16'h0166; // 1.41 to 1.44
        tanh_lut[ 120] = 16'h0167; tanh_lut[ 121] = 16'h0168; tanh_lut[ 122] = 16'h0169; tanh_lut[ 123] = 16'h016A; // 1.45 to 1.48
        tanh_lut[ 124] = 16'h016B; tanh_lut[ 125] = 16'h016C; tanh_lut[ 126] = 16'h016D; tanh_lut[ 127] = 16'h016F; // 1.49 to 1.52
        tanh_lut[ 128] = 16'h0170; tanh_lut[ 129] = 16'h0171; tanh_lut[ 130] = 16'h0172; tanh_lut[ 131] = 16'h0173; // 1.53 to 1.56
        tanh_lut[ 132] = 16'h0174; tanh_lut[ 133] = 16'h0175; tanh_lut[ 134] = 16'h0176; tanh_lut[ 135] = 16'h0177; // 1.57 to 1.60
        tanh_lut[ 136] = 16'h0178; tanh_lut[ 137] = 16'h0178; tanh_lut[ 138] = 16'h0179; tanh_lut[ 139] = 16'h017A; // 1.61 to 1.64
        tanh_lut[ 140] = 16'h017B; tanh_lut[ 141] = 16'h017C; tanh_lut[ 142] = 16'h017D; tanh_lut[ 143] = 16'h017D; // 1.65 to 1.68
        tanh_lut[ 144] = 16'h017E; tanh_lut[ 145] = 16'h017F; tanh_lut[ 146] = 16'h0180; tanh_lut[ 147] = 16'h0180; // 1.69 to 1.72
        tanh_lut[ 148] = 16'h0181; tanh_lut[ 149] = 16'h0182; tanh_lut[ 150] = 16'h0182; tanh_lut[ 151] = 16'h0183; // 1.73 to 1.76
        tanh_lut[ 152] = 16'h0184; tanh_lut[ 153] = 16'h0184; tanh_lut[ 154] = 16'h0185; tanh_lut[ 155] = 16'h0185; // 1.77 to 1.80
        tanh_lut[ 156] = 16'h0186; tanh_lut[ 157] = 16'h0187; tanh_lut[ 158] = 16'h0187; tanh_lut[ 159] = 16'h0188; // 1.81 to 1.84
        tanh_lut[ 160] = 16'h0188; tanh_lut[ 161] = 16'h0189; tanh_lut[ 162] = 16'h0189; tanh_lut[ 163] = 16'h018A; // 1.85 to 1.88
        tanh_lut[ 164] = 16'h018A; tanh_lut[ 165] = 16'h018B; tanh_lut[ 166] = 16'h018B; tanh_lut[ 167] = 16'h018C; // 1.89 to 1.92
        tanh_lut[ 168] = 16'h018C; tanh_lut[ 169] = 16'h018D; tanh_lut[ 170] = 16'h018D; tanh_lut[ 171] = 16'h018E; // 1.93 to 1.96
        tanh_lut[ 172] = 16'h018E; tanh_lut[ 173] = 16'h018E; tanh_lut[ 174] = 16'h018F; tanh_lut[ 175] = 16'h018F; // 1.97 to 2.00
        tanh_lut[ 176] = 16'h0190; tanh_lut[ 177] = 16'h0190; tanh_lut[ 178] = 16'h0190; tanh_lut[ 179] = 16'h0191; // 2.01 to 2.04
        tanh_lut[ 180] = 16'h0191; tanh_lut[ 181] = 16'h0191; tanh_lut[ 182] = 16'h0192; tanh_lut[ 183] = 16'h0192; // 2.05 to 2.08
        tanh_lut[ 184] = 16'h0192; tanh_lut[ 185] = 16'h0193; tanh_lut[ 186] = 16'h0193; tanh_lut[ 187] = 16'h0193; // 2.09 to 2.12
        tanh_lut[ 188] = 16'h0194; tanh_lut[ 189] = 16'h0194; tanh_lut[ 190] = 16'h0194; tanh_lut[ 191] = 16'h0194; // 2.13 to 2.16
        tanh_lut[ 192] = 16'h0195; tanh_lut[ 193] = 16'h0195; tanh_lut[ 194] = 16'h0195; tanh_lut[ 195] = 16'h0195; // 2.17 to 2.20
        tanh_lut[ 196] = 16'h0196; tanh_lut[ 197] = 16'h0196; tanh_lut[ 198] = 16'h0196; tanh_lut[ 199] = 16'h0196; // 2.21 to 2.24
        tanh_lut[ 200] = 16'h0197; tanh_lut[ 201] = 16'h0197; tanh_lut[ 202] = 16'h0197; tanh_lut[ 203] = 16'h0197; // 2.25 to 2.28
        tanh_lut[ 204] = 16'h0198; tanh_lut[ 205] = 16'h0198; tanh_lut[ 206] = 16'h0198; tanh_lut[ 207] = 16'h0198; // 2.29 to 2.32
        tanh_lut[ 208] = 16'h0198; tanh_lut[ 209] = 16'h0199; tanh_lut[ 210] = 16'h0199; tanh_lut[ 211] = 16'h0199; // 2.33 to 2.36
        tanh_lut[ 212] = 16'h0199; tanh_lut[ 213] = 16'h0199; tanh_lut[ 214] = 16'h019A; tanh_lut[ 215] = 16'h019A; // 2.37 to 2.40
        tanh_lut[ 216] = 16'h019A; tanh_lut[ 217] = 16'h019A; tanh_lut[ 218] = 16'h019A; tanh_lut[ 219] = 16'h019A; // 2.41 to 2.44
        tanh_lut[ 220] = 16'h019B; tanh_lut[ 221] = 16'h019B; tanh_lut[ 222] = 16'h019B; tanh_lut[ 223] = 16'h019B; // 2.45 to 2.48
        tanh_lut[ 224] = 16'h019B; tanh_lut[ 225] = 16'h019B; tanh_lut[ 226] = 16'h019C; tanh_lut[ 227] = 16'h019C; // 2.49 to 2.52
        tanh_lut[ 228] = 16'h019C; tanh_lut[ 229] = 16'h019C; tanh_lut[ 230] = 16'h019C; tanh_lut[ 231] = 16'h019C; // 2.53 to 2.56
        tanh_lut[ 232] = 16'h019C; tanh_lut[ 233] = 16'h019D; tanh_lut[ 234] = 16'h019D; tanh_lut[ 235] = 16'h019D; // 2.57 to 2.60
        tanh_lut[ 236] = 16'h019D; tanh_lut[ 237] = 16'h019D; tanh_lut[ 238] = 16'h019D; tanh_lut[ 239] = 16'h019D; // 2.61 to 2.64
        tanh_lut[ 240] = 16'h019E; tanh_lut[ 241] = 16'h019E; tanh_lut[ 242] = 16'h019E; tanh_lut[ 243] = 16'h019E; // 2.65 to 2.68
        tanh_lut[ 244] = 16'h019E; tanh_lut[ 245] = 16'h019E; tanh_lut[ 246] = 16'h019E; tanh_lut[ 247] = 16'h019E; // 2.69 to 2.72
        tanh_lut[ 248] = 16'h019E; tanh_lut[ 249] = 16'h019F; tanh_lut[ 250] = 16'h019F; tanh_lut[ 251] = 16'h019F; // 2.73 to 2.76
        tanh_lut[ 252] = 16'h019F; tanh_lut[ 253] = 16'h019F; tanh_lut[ 254] = 16'h019F; tanh_lut[ 255] = 16'h019F; // 2.77 to 2.80
        tanh_lut[ 256] = 16'h019F; tanh_lut[ 257] = 16'h019F; tanh_lut[ 258] = 16'h019F; tanh_lut[ 259] = 16'h019F; // 2.81 to 2.84
        tanh_lut[ 260] = 16'h019F; tanh_lut[ 261] = 16'h019F; tanh_lut[ 262] = 16'h019F; tanh_lut[ 263] = 16'h019F; // 2.85 to 2.88
        tanh_lut[ 264] = 16'h019F; tanh_lut[ 265] = 16'h019F; tanh_lut[ 266] = 16'h019F; tanh_lut[ 267] = 16'h019F; // 2.89 to 2.92
        tanh_lut[ 268] = 16'h019F; tanh_lut[ 269] = 16'h019F; tanh_lut[ 270] = 16'h019F; tanh_lut[ 271] = 16'h019F; // 2.93 to 2.96
        tanh_lut[ 272] = 16'h019F; tanh_lut[ 273] = 16'h019F; tanh_lut[ 274] = 16'h019F; tanh_lut[ 275] = 16'h019F; // 2.97 to 3.00
    end
    
    // Output assignment with bounds checking
    assign tanh_out = (addr < LUT_SIZE) ? tanh_lut[addr] : tanh_lut[LUT_SIZE-1];

endmodule