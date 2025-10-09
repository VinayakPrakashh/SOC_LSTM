// Input Gate Buffer
module buffer_i #(
    parameter DATA_WIDTH = 12,
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input we,
    input [ADDRESS_BITS-1:0] addr,
    input [DATA_WIDTH-1:0] din,
    output [DATA_WIDTH-1:0] dout
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem_array [0:(1<<ADDRESS_BITS)-1];

    // Initialize with Input gate linear values (S5.6 format)
    initial begin
        mem_array[0] = 12'h046;  // 1.1 = 1*64 + 0.1*64 = 70 = 0x046
        mem_array[1] = 12'h0C6;  // 3.1 = 3*64 + 0.1*64 = 198 = 0x0C6
        mem_array[2] = 12'h10D;  // 4.2 = 4*64 + 0.2*64 = 269 = 0x10D
        mem_array[3] = 12'h079;  // 1.9 = 1*64 + 0.9*64 = 121 = 0x079
    end

    // Write operation
    always @(posedge clk) begin
        if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[addr];
endmodule

// Forget Gate Buffer
module buffer_f #(
    parameter DATA_WIDTH = 12,
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input we,
    input [ADDRESS_BITS-1:0] addr,
    input [DATA_WIDTH-1:0] din,
    output  [DATA_WIDTH-1:0] dout
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem_array [0:(1<<ADDRESS_BITS)-1];

    // Initialize with Forget gate linear values (S5.6 format)
    initial begin
        mem_array[0] = 12'h0AB;  // 2.67 = 2*64 + 0.67*64 = 171 = 0x0AB
        mem_array[1] = 12'h01A;  // 0.41 = 0*64 + 0.41*64 = 26 = 0x01A
        mem_array[2] = 12'h356;  // 13.36 = 13*64 + 0.36*64 = 854 = 0x356
        mem_array[3] = 12'hFF8;  // -0.12 = -(0*64 + 0.12*64) = -8 = 0xFF8 (2's complement)
    end

    // Write operation
    always @(posedge clk) begin
        if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[addr];
endmodule

// Cell Gate Buffer
module buffer_c #(
    parameter DATA_WIDTH = 12,
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input we,
    input [ADDRESS_BITS-1:0] addr,
    input [DATA_WIDTH-1:0] din,
    output  [DATA_WIDTH-1:0] dout
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem_array [0:(1<<ADDRESS_BITS)-1];

    // Initialize with Cell gate linear values (S5.6 format)
    initial begin
        mem_array[0] = 12'h01F;  // 0.492 = 0*64 + 0.492*64 = 31 = 0x01F
        mem_array[1] = 12'h069;  // 1.638 = 1*64 + 0.638*64 = 105 = 0x069
        mem_array[2] = 12'h083;  // 2.042 = 2*64 + 0.042*64 = 131 = 0x083
        mem_array[3] = 12'h03C;  // 0.938 = 0*64 + 0.938*64 = 60 = 0x03C
    end

    // Write operation
    always @(posedge clk) begin
        if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[addr];
endmodule

// Output Gate Buffer
module buffer_o #(
    parameter DATA_WIDTH = 12,
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input we,
    input [ADDRESS_BITS-1:0] addr,
    input [DATA_WIDTH-1:0] din,
    output [DATA_WIDTH-1:0] dout
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem_array [0:(1<<ADDRESS_BITS)-1];

    // Initialize with Output gate linear values (S5.6 format)
    initial begin
        mem_array[0] = 12'h03F;  // 0.984 = 0*64 + 0.984*64 = 63 = 0x03F
        mem_array[1] = 12'h0A4;  // 2.556 = 2*64 + 0.556*64 = 164 = 0x0A4
        mem_array[2] = 12'h0DD;  // 3.464 = 3*64 + 0.464*64 = 221 = 0x0DD
        mem_array[3] = 12'h05C;  // 1.436 = 1*64 + 0.436*64 = 92 = 0x05C
    end

    // Write operation
    always @(posedge clk) begin
        if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[addr];
endmodule