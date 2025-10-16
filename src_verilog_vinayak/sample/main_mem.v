module main_mem #(
    parameter DATA_WIDTH = 12,
    parameter ADDR_BITS = 14,
    parameter DEPTH = 16*16

) (
    input clk,
    input rst,
    input we,
    input [DATA_WIDTH-1:0] data_in,
    input [ADDR_BITS-1:0] waddr,
    input [ADDR_BITS-1:0] raddr,
    output [DATA_WIDTH-1:0] data_out
);

reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
  // Fill first 16 elements of each of the first 16 rows
//integer i, row;
//initial begin
//    for (row = 0; row < 16; row = row + 1) begin
//        mem[row * 96 + 0]  = 12'b000000010000;  // +0.25   (16 in S5.6)
//        mem[row * 96 + 1]  = 12'b000000100000;  // +0.5    (32 in S5.6)
//        mem[row * 96 + 2]  = 12'b000000110000;  // +0.75   (48 in S5.6)
//        mem[row * 96 + 3]  = 12'b000001000000;  // +1.0    (64 in S5.6)
//        mem[row * 96 + 4]  = 12'b000001010000;  // +1.25   (80 in S5.6)
//        mem[row * 96 + 5]  = 12'b000001100000;  // +1.5    (96 in S5.6)
//        mem[row * 96 + 6]  = 12'b000001110000;  // +1.75   (112 in S5.6)
//        mem[row * 96 + 7]  = 12'b000010000000;  // +2.0    (128 in S5.6)
//        mem[row * 96 + 8]  = 12'b000010010000;  // +2.25   (144 in S5.6)
//        mem[row * 96 + 9]  = 12'b000010100000;  // +2.5    (160 in S5.6)
//        mem[row * 96 + 10] = 12'b000010110000;  // +2.75   (176 in S5.6)
//        mem[row * 96 + 11] = 12'b000011000000;  // +3.0    (192 in S5.6)
//        mem[row * 96 + 12] = 12'b000000001000;  // +0.125  (8 in S5.6)
//        mem[row * 96 + 13] = 12'b000000011000;  // +0.375  (24 in S5.6)
//        mem[row * 96 + 14] = 12'b000000101000;  // +0.625  (40 in S5.6)
//        mem[row * 96 + 15] = 12'b000000111000;  // +0.875  (56 in S5.6)
//    end
//    end
   initial begin
        $readmemh("readmem.mem", mem);
        $display("Weight memory initialized from file");
    end
always @(posedge clk) begin
    if (we) begin
        mem[waddr] <= data_in;
    end
end
assign data_out = mem[raddr];

endmodule