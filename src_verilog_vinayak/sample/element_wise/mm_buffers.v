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

    // Initialize with Input gate activated values (S5.6 format)
    initial begin
        mem_array[0] = 12'h030;  // 0.75026011 ≈ 0.75 = 0*64 + 0.75*64 = 48 = 0x030
        mem_array[1] = 12'h03D;  // 0.95689275 ≈ 0.96 = 0*64 + 0.96*64 = 61 = 0x03D
        mem_array[2] = 12'h03F;  // 0.98522597 ≈ 0.98 = 0*64 + 0.98*64 = 63 = 0x03F
        mem_array[3] = 12'h037;  // 0.86989153 ≈ 0.87 = 0*64 + 0.87*64 = 56 = 0x037
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

    // Initialize with Forget gate activated values (S5.6 format)
    initial begin
        mem_array[0] = 12'h03B;  // 0.93523303 ≈ 0.94 = 0*64 + 0.94*64 = 60 = 0x03C, but 59=0x03B
        mem_array[1] = 12'h026;  // 0.60108788 ≈ 0.60 = 0*64 + 0.60*64 = 38 = 0x026
        mem_array[2] = 12'h040;  // 0.99999842 ≈ 1.00 = 1*64 + 0.00*64 = 64 = 0x040
        mem_array[3] = 12'h01E;  // 0.47003595 ≈ 0.47 = 0*64 + 0.47*64 = 30 = 0x01E
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

    // Initialize with Cell gate activated values (S5.6 format)
    initial begin
        mem_array[0] = 12'h01D;  // 0.45580236 ≈ 0.46 = 0*64 + 0.46*64 = 29 = 0x01D
        mem_array[1] = 12'h03B;  // 0.92719246 ≈ 0.93 = 0*64 + 0.93*64 = 59 = 0x03B
        mem_array[2] = 12'h03E;  // 0.96687783 ≈ 0.97 = 0*64 + 0.97*64 = 62 = 0x03E
        mem_array[3] = 12'h02F;  // 0.734302   ≈ 0.73 = 0*64 + 0.73*64 = 47 = 0x02F
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

    // Initialize with Output gate activated values (S5.6 format)
    initial begin
        mem_array[0] = 12'h02E;  // 0.72790118 ≈ 0.73 = 0*64 + 0.73*64 = 47 = 0x02F, but 46=0x02E
        mem_array[1] = 12'h03B;  // 0.92797557 ≈ 0.93 = 0*64 + 0.93*64 = 59 = 0x03B
        mem_array[2] = 12'h03E;  // 0.96964592 ≈ 0.97 = 0*64 + 0.97*64 = 62 = 0x03E
        mem_array[3] = 12'h033;  // 0.80783446 ≈ 0.81 = 0*64 + 0.81*64 = 52 = 0x034, but 51=0x033
    end

    // Write operation
    always @(posedge clk) begin
        if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[addr];
endmodule