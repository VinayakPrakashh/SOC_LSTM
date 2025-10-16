// Cell state buffer (Ct) - for storing previous cell states
module buffer_ct #(
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

    // Memory array for cell state values
    reg [DATA_WIDTH-1:0] mem_array [0:(1<<ADDRESS_BITS)-1];

    // Initialize with previous cell state values (S5.6 format)
    // c_prev: [0.5 0.6 0.7 0.8]
    initial begin
        mem_array[0] = 12'h020;  // 0.5 = 0*64 + 0.5*64 = 32 = 0x020
        mem_array[1] = 12'h026;  // 0.6 = 0*64 + 0.6*64 = 38 = 0x026
        mem_array[2] = 12'h02D;  // 0.7 = 0*64 + 0.7*64 = 45 = 0x02D
        mem_array[3] = 12'h033;  // 0.8 = 0*64 + 0.8*64 = 51 = 0x033
    end

    // Write operation
    always @(posedge clk) begin
        if (rst) begin
            // Reset to initial values
            mem_array[0] <= 12'h020;  // 0.5
            mem_array[1] <= 12'h026;  // 0.6
            mem_array[2] <= 12'h02D;  // 0.7
            mem_array[3] <= 12'h033;  // 0.8
        end else if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[addr];
endmodule

// Hidden state buffer (ht) - for storing previous hidden states
module buffer_ht #(
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

    // Memory array for hidden state values
    reg [DATA_WIDTH-1:0] mem_array [0:(1<<ADDRESS_BITS)-1];

    // Initialize with previous hidden state values (S5.6 format)
    // h_prev: [0.1 0.2 0.3 0.4]
    initial begin
        mem_array[0] = 12'h006;  // 0.1 = 0*64 + 0.1*64 = 6 = 0x006
        mem_array[1] = 12'h00D;  // 0.2 = 0*64 + 0.2*64 = 13 = 0x00D
        mem_array[2] = 12'h013;  // 0.3 = 0*64 + 0.3*64 = 19 = 0x013
        mem_array[3] = 12'h01A;  // 0.4 = 0*64 + 0.4*64 = 26 = 0x01A
    end

    // Write operation
    always @(posedge clk) begin
        if (rst) begin
            // Reset to initial values
            mem_array[0] <= 12'h006;  // 0.1
            mem_array[1] <= 12'h00D;  // 0.2
            mem_array[2] <= 12'h013;  // 0.3
            mem_array[3] <= 12'h01A;  // 0.4
        end else if (we) begin
            mem_array[addr] <= din;
        end
    end

    assign dout = mem_array[addr];
endmodule