// Input Gate Buffer - Updated to S7.8 Format
module buffer_i #(
    parameter DATA_WIDTH = 16,  // Changed from 12 to 16 for S7.8
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input we,
    input [ADDRESS_BITS-1:0] addr,
    input [ADDRESS_BITS-1:0] raddr,
    input [DATA_WIDTH-1:0] din,
    output [DATA_WIDTH-1:0] dout
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem_array [0:(1<<ADDRESS_BITS)-1];

    // Initialize with Input gate linear values (S7.8 format)
    initial begin
        mem_array[0] = 16'h011A;  // 1.1 = 1*256 + 0.1*256 = 281 = 0x011A
        mem_array[1] = 16'h031A;  // 3.1 = 3*256 + 0.1*256 = 793 = 0x031A
        mem_array[2] = 16'h0433;  // 4.2 = 4*256 + 0.2*256 = 1075 = 0x0433
        mem_array[3] = 16'h01E6;  // 1.9 = 1*256 + 0.9*256 = 486 = 0x01E6
    end

    // Write operation
    always @(posedge clk) begin
        if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[raddr];
endmodule

// Forget Gate Buffer - Updated to S7.8 Format
module buffer_f #(
    parameter DATA_WIDTH = 16,  // Changed from 12 to 16 for S7.8
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input we,
    input [ADDRESS_BITS-1:0] addr,
    input [ADDRESS_BITS-1:0] raddr,
    input [DATA_WIDTH-1:0] din,
    output [DATA_WIDTH-1:0] dout
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem_array [0:(1<<ADDRESS_BITS)-1];

    // Initialize with Forget gate linear values (S7.8 format)
    initial begin
        mem_array[0] = 16'h02AB;  // 2.67 = 2*256 + 0.67*256 = 683 = 0x02AB
        mem_array[1] = 16'h0069;  // 0.41 = 0*256 + 0.41*256 = 105 = 0x0069
        mem_array[2] = 16'h0D5C;  // 13.36 = 13*256 + 0.36*256 = 3420 = 0x0D5C
        mem_array[3] = 16'h801F;  // -0.12 = sign bit(1) + 0.12*256 = 31 with sign = 0x801F
    end

    // Write operation
    always @(posedge clk) begin
        if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[raddr];
endmodule

// Cell Gate Buffer - Updated to S7.8 Format
module buffer_g #(
    parameter DATA_WIDTH = 16,  // Changed from 12 to 16 for S7.8
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input we,
    input [ADDRESS_BITS-1:0] addr,
    input [ADDRESS_BITS-1:0] raddr,
    input [DATA_WIDTH-1:0] din,
    output [DATA_WIDTH-1:0] dout
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem_array [0:(1<<ADDRESS_BITS)-1];

    // Initialize with Cell gate linear values (S7.8 format)
    initial begin
        mem_array[0] = 16'h007E;  // 0.492 = 0*256 + 0.492*256 = 126 = 0x007E
        mem_array[1] = 16'h01A3;  // 1.638 = 1*256 + 0.638*256 = 419 = 0x01A3
        mem_array[2] = 16'h020B;  // 2.042 = 2*256 + 0.042*256 = 523 = 0x020B
        mem_array[3] = 16'h00F0;  // 0.938 = 0*256 + 0.938*256 = 240 = 0x00F0
    end

    // Write operation
    always @(posedge clk) begin
        if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[raddr];
endmodule

// Output Gate Buffer - Updated to S7.8 Format
module buffer_o #(
    parameter DATA_WIDTH = 16,  // Changed from 12 to 16 for S7.8
    parameter ADDRESS_BITS = 2
) (
    input clk,
    input rst,
    input we,
    input [ADDRESS_BITS-1:0] addr,
    input [ADDRESS_BITS-1:0] raddr,
    input [DATA_WIDTH-1:0] din,
    output [DATA_WIDTH-1:0] dout
);

    // Memory array
    reg [DATA_WIDTH-1:0] mem_array [0:(1<<ADDRESS_BITS)-1];

    // Initialize with Output gate linear values (S7.8 format)
    initial begin
        mem_array[0] = 16'h00FC;  // 0.984 = 0*256 + 0.984*256 = 252 = 0x00FC
        mem_array[1] = 16'h028F;  // 2.556 = 2*256 + 0.556*256 = 655 = 0x028F
        mem_array[2] = 16'h0377;  // 3.464 = 3*256 + 0.464*256 = 887 = 0x0377
        mem_array[3] = 16'h016F;  // 1.436 = 1*256 + 0.436*256 = 367 = 0x016F
    end

    // Write operation
    always @(posedge clk) begin
        if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[raddr];
endmodule