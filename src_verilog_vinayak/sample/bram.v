module weight_bram #(
    parameter DATA_WIDTH = 12,
    parameter ADDR_WIDTH = 10,
    parameter DEPTH = 564
) (
    input clk,
    input rst,
    input [ADDR_WIDTH-1:0] addr_0,
    output [DATA_WIDTH-1:0] data_out_0,
    input [ADDR_WIDTH-1:0] addr_1,
    output [DATA_WIDTH-1:0] data_out_1,
    input [ADDR_WIDTH-1:0] addr_2,
    output [DATA_WIDTH-1:0] data_out_2,
    input [ADDR_WIDTH-1:0] addr_3,
    output [DATA_WIDTH-1:0] data_out_3,
    input [ADDR_WIDTH-1:0] addr_4,
    output [DATA_WIDTH-1:0] data_out_4,
    input [ADDR_WIDTH-1:0] addr_5,
    output [DATA_WIDTH-1:0] data_out_5,
    input [ADDR_WIDTH-1:0] addr_6,
    output [DATA_WIDTH-1:0] data_out_6,
    input [ADDR_WIDTH-1:0] addr_7,
    output [DATA_WIDTH-1:0] data_out_7
    input [ADDR_WIDTH-1:0] addr_8,
    output [DATA_WIDTH-1:0] data_out_8,
    input [ADDR_WIDTH-1:0] addr_9,
    output [DATA_WIDTH-1:0] data_out_9,
    input [ADDR_WIDTH-1:0] addr_10,
    output [DATA_WIDTH-1:0] data_out_10,
    input [ADDR_WIDTH-1:0] addr_11,
    output [DATA_WIDTH-1:0] data_out_11,
    input [ADDR_WIDTH-1:0] addr_12,
    output [DATA_WIDTH-1:0] data_out_12,
    input [ADDR_WIDTH-1:0] addr_13,
    output [DATA_WIDTH-1:0] data_out_13,
    input [ADDR_WIDTH-1:0] addr_14,
    output [DATA_WIDTH-1:0] data_out_14,
    input [ADDR_WIDTH-1:0] addr_15,
    output [DATA_WIDTH-1:0] data_out_15
    
);
    reg [DATA_WIDTH-1:0] bram_0 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_1 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_2 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_3 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_4 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_5 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_6 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_7 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_8 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_9 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_10 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_11 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_12 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_13 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_14 [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] bram_15 [0:DEPTH-1];

    initial begin
        $readmemh("weights_0.mem", bram_0);
        $readmemh("weights_1.mem", bram_1);
        $readmemh("weights_2.mem", bram_2);
        $readmemh("weights_3.mem", bram_3);
        $readmemh("weights_4.mem", bram_4);
        $readmemh("weights_5.mem", bram_5);
        $readmemh("weights_6.mem", bram_6);
        $readmemh("weights_7.mem", bram_7);
        $readmemh("weights_8.mem", bram_8);
        $readmemh("weights_9.mem", bram_9);
        $readmemh("weights_10.mem", bram_10);
        $readmemh("weights_11.mem", bram_11);
        $readmemh("weights_12.mem", bram_12);
        $readmemh("weights_13.mem", bram_13);
        $readmemh("weights_14.mem", bram_14);
        $readmemh("weights_15.mem", bram_15);
    end

assign data_out_0 = bram_0[addr_0];
assign data_out_1 = bram_1[addr_1];
assign data_out_2 = bram_2[addr_2];
assign data_out_3 = bram_3[addr_3];
assign data_out_4 = bram_4[addr_4];
assign data_out_5 = bram_5[addr_5];
assign data_out_6 = bram_6[addr_6];
assign data_out_7 = bram_7[addr_7];
assign data_out_8 = bram_8[addr_8];
assign data_out_9 = bram_9[addr_9];
assign data_out_10 = bram_10[addr_10];
assign data_out_11 = bram_11[addr_11];
assign data_out_12 = bram_12[addr_12];
assign data_out_13 = bram_13[addr_13];
assign data_out_14 = bram_14[addr_14];
assign data_out_15 = bram_15[addr_15];

endmodule